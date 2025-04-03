import numpy as np
import os
from motion_pred.utils.dataset import Dataset
from motion_pred.utils.skeleton import Skeleton

import pdb

""" 
SMPL (HumanML3D) 
"""
SMPL_JOINT_ORDER = [
'Root',
'Left Hip', 'Right Hip',
'Spine1',
'Left Knee','Right Knee',
'Spine2',
'Left Ankle', 'Right Ankle',
'Spine3',
'Left Foot', 'Right Foot',
'Neck', 
'Left Collar', 'Right Collar',
'Head',
'Left Shoulder', 'Right Shoulder',
'Left Elbow', 'Right Elbow',
'Left Wrist', 'Right Wrist',
'Prop'
]

SMPL_JOINT_PAIRS = [
    ("Head", "Neck"),
    ("Neck", "Spine3"),
    ("Spine3", "Spine2"),
    ("Spine2", "Spine1"),
    ("Spine1", "Root"),
    ("Root", "Left Hip"), 
    ("Root", "Right Hip"),
    ("Left Hip", "Left Knee"),
    ("Left Knee", "Left Ankle"),
    ("Left Ankle", "Left Foot"),
    ("Right Hip", "Right Knee"),
    ("Right Knee", "Right Ankle"),
    ("Right Ankle", "Right Foot"), 
    ("Spine3", "Left Collar"),    
    ("Spine3", "Right Collar"),
    ("Left Collar", "Left Shoulder"),
    ("Left Shoulder", "Left Elbow"),    
    ("Left Elbow", "Left Wrist"),    
    ("Right Collar", "Right Shoulder"),
    ("Right Shoulder", "Right Elbow"),    
    ("Right Elbow", "Right Wrist")
]

class DatasetSMPL_ours(Dataset):

    def __init__(self, mode, t_his=10, t_pred=190, actions='all', use_vel=False):
        self.use_vel = use_vel
        self.smpl_joint_order = SMPL_JOINT_ORDER
        self.smpl_joint_pairs = SMPL_JOINT_PAIRS
        self.max_motion_len = 199
        super().__init__(mode, t_his, t_pred, actions)
        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):

        DATAROOT = '/data2/heejae/Project/TextToMotion/AVA_challenge_Correct_GT/Baseline/CVAE_baseline/data'
        self.skel_file = os.path.join(f'{DATAROOT}/dataset_motion_smplID_20fps_avasplit.npz')
        self.text_file = os.path.join(f'{DATAROOT}/dataset_text_high_avasplit.npz')

        self.subjects_split = {'train': ['s_train'],
                               'test': ['s_test']}
        
        self.subjects = ['%s' % x for x in self.subjects_split[self.mode]]

        joint_child_from_connection = [SMPL_JOINT_ORDER.index(SMPL_JOINT_PAIRS[i][1]) for i in range(len(SMPL_JOINT_PAIRS))]
        joint_child_names = [SMPL_JOINT_ORDER[i] for i in joint_child_from_connection]
        joint_parent_from_connection = [SMPL_JOINT_ORDER.index(SMPL_JOINT_PAIRS[i][0]) for i in range(len(SMPL_JOINT_PAIRS))]
        joint_parent_names = [SMPL_JOINT_ORDER[i] for i in joint_parent_from_connection]

        joint_p = [SMPL_JOINT_ORDER.index(joint_parent_names[joint_child_names.index(SMPL_JOINT_ORDER[i])])  if SMPL_JOINT_ORDER[i] in joint_child_names else -1 for i in range(len(SMPL_JOINT_ORDER))]
        joint_l = [SMPL_JOINT_ORDER.index(SMPL_JOINT_ORDER[i]) for i in range(len(SMPL_JOINT_ORDER)) if 'Left' in SMPL_JOINT_ORDER[i]]
        joint_r = [SMPL_JOINT_ORDER.index(SMPL_JOINT_ORDER[i]) for i in range(len(SMPL_JOINT_ORDER)) if 'Right' in SMPL_JOINT_ORDER[i]]        
        
        self.skeleton = Skeleton(parents = joint_p,
                                 joints_left = joint_l,
                                 joints_right = joint_r)
        self.removed_joints = {}

        self.kept_joints = np.array([x for x in range(len(SMPL_JOINT_ORDER))])

        self.skeleton.remove_joints(self.removed_joints)
        self.process_data()

    def process_data(self):
        
        data_o_skel = np.load(self.skel_file, allow_pickle=True)['data'].item()        
        data_f_skel = dict(filter(lambda x:x[0] in self.subjects, data_o_skel.items()))
        
        data_o_text = np.load(self.text_file, allow_pickle=True)['data'].item()        
        data_f_text = dict(filter(lambda x:x[0] in self.subjects, data_o_text.items()))
        
        all_motion3d = []
        sub_list = list(data_f_skel.keys())
        for sub_name in sub_list:
            data_skel = data_f_skel[sub_name]
            data_text = data_f_text[sub_name]
            
            action_to_remove = []
            for action in data_skel.keys():
                # normalize motion data 
                motion3d = data_skel[action] - np.expand_dims(data_skel[action][:,0,:], axis=1) 
                motion3d = motion3d[:self.max_motion_len,1:,:] 
                data_skel[action] = motion3d
                all_motion3d.append(motion3d) # all sequence w/o root
                
                # text data
                text_embd = data_text[action]
                data_text[action] = text_embd
                
                if motion3d.shape[0] != self.max_motion_len: # if not 200 length, remove
                    action_to_remove.append(action)

            for key_to_remove in action_to_remove:
                del data_skel[key_to_remove]
                del data_text[key_to_remove]
                    
        all_motion3d = np.concatenate(all_motion3d)
        self.mean_3d = all_motion3d.mean(axis=0)
        self.std_3d = all_motion3d.std(axis=0)

        self.data_skeleton = data_f_skel
        self.data_text_embd = data_f_text


