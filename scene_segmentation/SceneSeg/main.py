import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
import os
import pandas as pd
import json
import argparse
import random
import time
from sklearn.metrics import average_precision_score
import shutil
import os.path as osp
from BiLSTM_protocol import BiLSTM
from movienet_seg_data import MovieNet_SceneSeg_Dataset_Embeddings_Val

def main(args):
    setup_seed(100)
    model = BiLSTM(
        input_feature_dim=args.dim,
        input_drop_rate=args.input_drop_rate
    ).cuda()

    # trained parameter 불러오기
    checkpoint = torch.load(args.model_path)   

    # to test trained model parameter
    # print(list(checkpoint['state_dict'].keys()) == list(model.state_dict().keys()))
    # output : True

    # trained parameter를 model에 적용
    model.load_state_dict(checkpoint['state_dict'])

    label_weights = torch.Tensor([args.loss_weight[0], args.loss_weight[1]]).cuda()
    criterion = nn.CrossEntropyLoss(label_weights).cuda()

    optimizer = torch.optim.SGD(model.parameters(), 
        args.lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    dataset = MovieNet_SceneSeg_Dataset_Embeddings_Val(
        pkl_path=args.pkl_path,
        sampled_shot_num=args.seq_len
    )

    # shuffle = False
    loader = torch.utils.data.DataLoader(dataset, args.test_bs, num_workers=args.workers,
        shuffle=False, pin_memory=True, drop_last=False)
    
    inference(args, model, loader) # 파일로 저장까지 완료

@torch.no_grad()
def inference(args, model, loader, threshhold=0.5):
    model.eval()
    stride = args.seq_len // 2
    result_all = {}
    for batch_idx, (data, imdb, num_shot) in enumerate(loader):
        imdb = imdb[0]
        num_shot = num_shot[0]
        result_all[imdb] = {} 
        data = data.view(-1, args.dim).cuda(non_blocking=True)
        data_len = data.size(0)
        gt_len = num_shot
        prob_all = []
        for w_id in range(data_len//stride):
            start_pos = w_id*stride
            _data = data[start_pos:start_pos + args.seq_len].unsqueeze(0)
            output = model(_data, None)
            output = output.view(-1, 2)
            prob = output[:, 1]
            prob = prob[stride//2:stride+stride//2].squeeze()
            prob_all.append(prob.cpu())

        preb_all = torch.cat(prob_all,axis=0)[:gt_len].numpy()
        pre = np.nan_to_num(preb_all) > threshhold
        pre = pre.astype(int)

        # json에 time 정보까지 저장하는 옵션
        if args.get_time:
            result_all = get_result_with_time(args, pre, result_all, imdb)
        else: 
            result_all = get_result(args, pre, result_all, imdb)

    tmp = "/home/previ01/changjun/vmr_baseline/scene_segmentation/vss_result/vss_result.json"

    with open(tmp, 'w', encoding='utf-8') as file:
        json.dump(result_all, file, indent="\t")


def get_result(args, pre, result_all, imdb):

    frame_info_path = os.path.join(args.frame_info_path, f"{imdb}.txt")
    frame_info = pd.read_csv(frame_info_path, header=None, sep='\s+')

    scene_cnt = 0
    is_first = True

    for i, shot_pre in enumerate(pre):
        if is_first:
            result_all[imdb][str(scene_cnt)] = {"shot":[], "frame":[str(frame_info.iloc[i, 0])]}
            is_first = False
        result_all[imdb][str(scene_cnt)]["shot"].append(str(i).zfill(4))
        if shot_pre == 1 or i == (len(pre) - 1):
            result_all[imdb][str(scene_cnt)]["frame"].append(str(frame_info.iloc[i, 1]))
            is_first = True
            scene_cnt += 1

    return result_all


def get_result_with_time(args, pre, result_all, imdb):

    frame_info_path = os.path.join(args.frame_info_path, f"{imdb}.txt")
    frame_info = pd.read_csv(frame_info_path, header=None, sep='\s+')

    time_info_path = os.path.join(args.time_info_path, f"{imdb}.csv")
    time_info = pd.read_csv(time_info_path, skiprows=2, header=None)
    time_info = [0.]+list(time_info.iloc[:, 1])
    time_info = [round(num, 2) for num in time_info]    

    scene_cnt = 0
    is_first = True

    for i, shot_pre in enumerate(pre):
        if is_first:
            result_all[imdb][str(scene_cnt)] = {"shot":[], "frame":[str(frame_info.iloc[i, 0])], "time":[float(time_info[frame_info.iloc[i, 0]])]}
            is_first = False
        result_all[imdb][str(scene_cnt)]["shot"].append(str(i).zfill(4))
        if shot_pre == 1 or i == (len(pre) - 1):
            result_all[imdb][str(scene_cnt)]["frame"].append(str(frame_info.iloc[i, 1]))
            result_all[imdb][str(scene_cnt)]["time"].append(float(time_info[frame_info.iloc[i, 1]]))
            is_first = True
            scene_cnt += 1

    return result_all 
    

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def set_log(args):
    time_str = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    
    args.log_file = './output/log_' + time_str + '.txt'
    args.save_dir = args.save_dir + 'seg_checkpoints/' + time_str + '/'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists('./output/'):
        os.makedirs('./output/')

def to_log(args, content, echo=False):
    with open(args.log_file, 'a') as f:
        f.writelines(content+'\n')
    if echo:
        print(content)

def adjust_learning_rate(args, optimizer, epoch):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_ap(gts_raw,preds_raw,is_list=True):
    if is_list:
        gts,preds = [],[]
        for gt_raw in gts_raw:
            gts.extend(gt_raw.tolist())
        for pred_raw in preds_raw:
            preds.extend(pred_raw.tolist())
    else: 
        gts = np.array(gts_raw)
        preds = np.array(preds_raw)
    # print ("AP ",average_precision_score(gts, preds))
    return average_precision_score(np.nan_to_num(gts), np.nan_to_num(preds))
    # return average_precision_score(gts, preds)

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    os.makedirs(osp.dirname(fpath),exist_ok=True)
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def get_config():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # important argument
    parser.add_argument('--get_time', action="store_true")
    parser.add_argument('--frame_info_path', default = "/home/previ01/changjun/vmr_baseline/shot_detection/shot_detection_result/shot_txt", type = str)
    parser.add_argument('--time_info_path', default = "/home/previ01/changjun/vmr_baseline/shot_detection/shot_detection_result/shot_stats", type = str)
    parser.add_argument('--model_path', default = "/home/previ01/changjun/vmr_baseline/scene_segmentation/trained_models/model_best.pth.tar", type = str)
    parser.add_argument('--pkl-path', default="/home/previ01/changjun/vmr_baseline/scene_segmentation/embeddings/2023-07-27_14_34_12/embedding_result.pkl", type=str,
                    help='the path of pickle data')

    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--train-bs', default=12, type=int)
    parser.add_argument('--test-bs', default=1, type=int)
    parser.add_argument('--shot-num', default=10, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--gpu-id', type=str, default='0', help='gpu id')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay', dest='weight_decay')
    
    parser.add_argument('--save-dir', default="./output/", type=str,
                        help='the path of checkpoints')
    # parser.add_argument('--json-dir', default="/home/previ01/changjun/vmr_baseline/scene_segmentation/vss_result/test_result.json", type=str)

    # loss weight
    parser.add_argument('--loss-weight', default=[1, 4], nargs='+', type=float,
                    help='loss weight')
    parser.add_argument('--sample-shulle-rate', default=1.0, type=float)
    parser.add_argument('--input-drop-rate', default=0.2, type=float)
    # lr schedule
    parser.add_argument('--schedule', default=[160, 180], nargs='+',
                    help='learning rate schedule (when to drop lr by a ratio)')

    parser.add_argument('-j', '--workers', default=16, type=int,
                        help='number of workers')
    parser.add_argument('--dim', default=2048, type=int)
    parser.add_argument('--seq-len', default=40, type=int)
    parser.add_argument('--test-interval', default=1, type=int)
    parser.add_argument('--test-milestone', default=100, type=int)

    args = parser.parse_args()

    # assert
    assert args.seq_len % 4 == 0

    # select GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    set_log(args)
    for arg in vars(args):
        to_log(args,arg.ljust(20)+':'+str(getattr(args, arg)), True)  
    return args

if __name__ == '__main__':
    args = get_config()
    main(args)