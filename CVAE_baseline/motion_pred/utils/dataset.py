import numpy as np
import pdb

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Dataset:

    def __init__(self, mode, t_his, t_pred, actions='all'):
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.prepare_data()
        self.std, self.mean = None, None
        self.data_len = sum([seq.shape[0] for data_s in self.data_skeleton.values() for seq in data_s.values()])
        self.traj_dim = (self.kept_joints.shape[0] - 1) * 3
        self.normalized = False
        # iterator specific
        self.sample_ind = None

    def prepare_data(self):
        raise NotImplementedError

    def sample(self):
        subject = np.random.choice(self.subjects)

        dict_skel = self.data_skeleton[subject]    
        dict_text = self.data_text_embd[subject]  

        action = np.random.choice(list(dict_skel.keys()))
   
        motion3d = dict_skel[action]
        text_embd = dict_text[action]

        return motion3d[None, ...], text_embd[None, ...]

    def sampling_generator(self, num_samples=1000, batch_size=8):
        for i in range(num_samples // batch_size):
            sample_skel = []
            sample_text = []

            for i in range(batch_size):
                sample_skel_i, sample_text_i = self.sample()
                sample_skel.append(sample_skel_i)
                sample_text.append(sample_text_i)

            sample_skel = np.concatenate(sample_skel, axis=0)
            sample_text = np.concatenate(sample_text, axis=0)

            yield sample_skel, sample_text

    def iter_generator(self, step=25):

        s_list = list(self.data_skeleton.keys())
        for s_name in s_list:
            data_s_skel = self.data_skeleton[s_name]
            data_s_text = self.data_text_embd[s_name]

            seq_list = list(data_s_skel.keys())
            for seq_name in seq_list:
                motion3d = data_s_skel[seq_name]
                text_embd = data_s_text[seq_name]
                
                yield [motion3d[None, ...], text_embd[None, ...], seq_name]
                
 

