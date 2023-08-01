import pickle
import torch
import torch.utils.data as data
import numpy as np
import random

class MovieNet_SceneSeg_Dataset_Embeddings_Val(data.Dataset):
    def __init__(self, pkl_path, frame_size=3, shot_num=1, 
        sampled_shot_num=100):
        self.shot_num = shot_num
        self.pkl_path = pkl_path
        self.frame_size = frame_size
        self.sampled_shot_num = sampled_shot_num
        self.dict_idx_shot = {}
        self.data_length = 0
        fileObject = open(self.pkl_path, 'rb')
        self.pickle_data = pickle.load(fileObject)
        fileObject.close()
        self.total_video_num = len(self.pickle_data.keys())
        idx = 0
        for k, v in self.pickle_data.items():
            self.dict_idx_shot[idx] = (k, v)
            idx += 1
        print(f'video num: {self.total_video_num}')
        self.data_length = idx

    def _padding(self, data):
        stride = self.sampled_shot_num // 2
        shot_len = data.size(0)
        p_l = data[0].repeat(self.sampled_shot_num // 4, 1)
        p_r_len = self.sampled_shot_num // 4
        res = shot_len % (stride)
        if res != 0:
            p_r_len += (stride) - res
        p_r = data[-1].repeat(p_r_len, 1)
        pad_data = torch.cat((p_l, data, p_r),0)
        assert pad_data.size(0) % stride == 0
        return pad_data

    def __getitem__(self, idx):
        k, v = self.dict_idx_shot[idx]
        num_shot = len(v)
        data = np.array([v[i][0] for i in range(num_shot)])
        data = torch.from_numpy(data).squeeze(1)
        data = self._padding(data)
        return data, k, num_shot

    def __len__(self):
        return self.data_length