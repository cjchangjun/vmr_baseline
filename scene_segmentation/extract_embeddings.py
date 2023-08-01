import pickle
import os
import torch
import argparse
import time
from models.backbones.visual.resnet import encoder_resnet50
import json
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader

# ActivityNet_SingleShot_Dataset
class MovieNet_SingleShot_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_path, shot_info_path, transform,
        frame_per_shot = 3):

        # img_path : 각 video의 key frame들이 담긴 폴더들이 있는 폴더의 경로
        self.img_path = img_path

        self.frame_per_shot = frame_per_shot
        self.transform = transform
        self.idx_imdb_map = {}
        data_length = 0
        
        # video_id : ex) "tt204856" or "v_4RdaAe43"
        # shot_id  : ex) "0000" or "0023"

        video_ids = os.listdir(self.img_path)
        for video_id in video_ids:
            frames = os.listdir(os.path.join(self.img_path, video_id))
            shot_ids = [frame.split('.')[0].split('_')[1] for frame in frames]
            shot_ids = list(set(shot_ids)) 
            shot_ids.sort() # it is critical!!!
            for shot_id in shot_ids:
                self.idx_imdb_map[data_length] = (video_id, shot_id)
                data_length += 1

    def __len__(self):
        return len(self.idx_imdb_map.keys())

    def _process(self, idx):

        # data : video_id, shot_id에 해당하는 shot의 frame들을 전처리한 실제 data
        # imdb : video_id
        # _id  : shot_id
        imdb, _id  = self.idx_imdb_map[idx]
        img_path_0 =  f'{self.img_path}/{imdb}/shot_{_id}_img_0.jpg'
        img_path_1 =  f'{self.img_path}/{imdb}/shot_{_id}_img_1.jpg'
        img_path_2 =  f'{self.img_path}/{imdb}/shot_{_id}_img_2.jpg'
        img_0      = cv2.cvtColor(cv2.imread(img_path_0), cv2.COLOR_BGR2RGB)
        img_1      = cv2.cvtColor(cv2.imread(img_path_1), cv2.COLOR_BGR2RGB)
        img_2      = cv2.cvtColor(cv2.imread(img_path_2), cv2.COLOR_BGR2RGB)
        data_0     = self.transform(img_0)
        data_1     = self.transform(img_1)
        data_2     = self.transform(img_2)
        data       = torch.cat([data_0, data_1, data_2], axis=0)
  
        return data, (imdb, _id)


    def __getitem__(self, idx):
        return self._process(idx)

def get_loader(cfg):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    _transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])
    dataset = MovieNet_SingleShot_Dataset(
        img_path = cfg.shot_img_path,
        shot_info_path = cfg.shot_info_path,
        transform = _transform,
        frame_per_shot = cfg.frame_per_shot
    )

    # shuffle=False : 순서대로 넣어주기 위함 (중요)
    loader = DataLoader(
        dataset, batch_size=cfg.bs,  drop_last=False,
        shuffle=False, num_workers=cfg.worker_num, pin_memory=True
    )
    return loader

def get_encoder(model_name='resnet50', weight_path='', input_channel=9):
    encoder = None
    model_name = model_name.lower()
    if model_name == 'resnet50':

        encoder = encoder_resnet50(weight_path='',input_channel=input_channel)
        model_weight = torch.load(weight_path,map_location=torch.device('cpu'))['state_dict']
        pretrained_dict = {}

        # model_weight dictionary에는 query encoder와 key encoder의 trained parameter가 있는데,
        # 우리는 inference에서 query encoder만 필요하므로, pretrained_dict에 query encoder의 parameter만 모아서 저장한다.
        for k, v in model_weight.items():
            # moco loading 
            if k.startswith('module.encoder_k'):
                continue
            if k == 'module.queue' or k == 'module.queue_ptr':
                continue
            # fc는 inference에서는 사용하지 않으므로 굳이 저장하지 않음.
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                k = k[17:]
            
            pretrained_dict[k] = v

        # strict = False -> 일치하지 않는 키들은 무시하도록 설정
        encoder.load_state_dict(pretrained_dict, strict = False)

        print(f'loaded from {weight_path}')
    return encoder


@torch.no_grad()
def get_save_embeddings(model, loader, shot_num, filename, log_interval=100):
    # dict
    # key: index, value: [(embeddings, label), ...]
    embeddings = {} 
    model.eval()
    
    print(f'total length of dataset: {len(loader.dataset)}')
    print(f'total length of loader: {len(loader)}')
    
    # index : [tensor of video_id in 0 dim, tensor of shot_id in 0 dim]
    for batch_idx, (data, index) in enumerate(loader):
        if batch_idx % log_interval == 0:
            print(f'processed: {batch_idx}')
        
        data = data.cuda(non_blocking=True) # ([bs, shot_num, 9, 224, 224])
        data = data.view(-1, 9, 224, 224)

        # model에 False를 넘겨줌 -> 마지막 fc layer는 수행하지 않음.
        output = model(data, False)   # ([bs * shot_num, 2048])

        # key : video_id
        for i, key in enumerate(index[0]):
            if key not in embeddings:
                embeddings[key] = []
            t_emb = output[i*shot_num:(i+1)*shot_num].cpu().numpy()
            embeddings[key].append(t_emb.copy())
    pickle.dump(embeddings, open(filename, 'wb'))


def extract_features(cfg):
    time_str = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    save_dir = os.path.join(cfg.save_dir, time_str)
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    encoder = get_encoder(
        model_name=cfg.model_name,
        weight_path=cfg.model_path,
        input_channel=cfg.frame_per_shot * 3
        ).cuda()
    
    loader = get_loader(cfg)
    filename = os.path.join(save_dir, 'embedding_result.pkl')
    get_save_embeddings(encoder, 
        loader, 
        cfg.shot_num, 
        filename, 
        log_interval=100
    )


def to_log(cfg, content, echo=True):
    with open(cfg.log_file, 'a') as f:
        f.writelines(content+'\n')
    if echo: print(content)


def get_config():

    parser = argparse.ArgumentParser()

    # important argument
    parser.add_argument('--shot_img_path', type=str, default='/home/previ01/changjun/vmr_baseline/shot_detection/shot_detection_result/shot_keyf/')
    parser.add_argument('--model_path', type=str, default='/home/previ01/changjun/vmr_baseline/scene_segmentation/trained_models/checkpoint_0099.pth.tar')
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--save_dir', type=str, default='./embeddings/')    

    parser.add_argument('--shot_info_path', type=str, 
        default='./data/movie1K.scene_seg_318_name_index_shotnum_label.v1.json')
    parser.add_argument('--Type', type=str, default='train', choices=['train','test','val','all'])
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--frame_per_shot', type=int, default=3)
    parser.add_argument('--shot_num', type=int, default=1)
    parser.add_argument('--worker_num', type=int, default=16)
    parser.add_argument('--gpu-id', type=str, default='0')
    cfg = parser.parse_args()

    # select GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id

    return cfg


if __name__ == '__main__':
    cfg = get_config()
    extract_features(cfg)