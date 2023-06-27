import os
import random

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# from data_utils import read_midi, read_csv


class MidiMotionDataSet(Dataset):
    def __init__(self, file_list_txt):  # 定義路徑，建立 file list # , midi_sr, motion_sr
        self.read_data = {}
        self.read_data["midi"] = []
        self.read_data["motion"] = []
        self.dataset_len = 0

        with open(file_list_txt, 'r') as file:
            lines = file.readlines()

        piece_count = 0
        # already done for preprocess
        for line in lines:  # midi_list replace 'midi' with 'motion' then get motion data.

            midi_file_path = "./" + line.rstrip()
            # replace all "midi" with "motion"
            motion_file_path = "./" + line.rstrip().replace("midi", "motion")

            self.read_data["midi"].append(midi_file_path)
            self.read_data["motion"].append(motion_file_path)
            piece_count += 1

        self.dataset_len = piece_count * 100 #2200
        self.performer_namecodes = self.read_data['midi']
        # print(self.performer_namecodes)
        for performer_piece in self.performer_namecodes:
            self.music_files_index = {
                performer_piece: index for index, file_path in enumerate(self.read_data["midi"])}
        print("dataset_len", self.dataset_len)

    def __len__(self): # len / batch_size = 1 epoch have how many batch
        return self.dataset_len #index need to fit length

    def __getitem__(self, index):  # 這裡才要讀資料
        # load_pickle with index
        # print("index", index)
        midi_data_input = open(self.read_data["midi"][index%22], 'rb')
        motion_data_input = open(self.read_data["motion"][index%22], 'rb')

        midi_data = pickle.load(midi_data_input)
        motion_data = pickle.load(motion_data_input)

        midi_data_input.close()
        motion_data_input.close()
        
        # TODO: 1. Should concat all data to one array, and then split to sequence_len=512 segment.
        # TODO:     >> If sequence_len cannot divide by 512, then eliminate it.
        # TODO: 2. May use sliding window to split more segment.
        window_size = 512
        
        left_edge = random.randint(0, len(midi_data) - window_size)
        # print(left_edge)
        segment_midi = np.array(midi_data[left_edge:left_edge+window_size])
        segment_motion =  np.array(motion_data[left_edge:left_edge+window_size])
        # print("segment_midi", segment_midi.shape)
        # print("segment_motion", segment_motion.shape)
        
        # midi_data = midi_data[None, :]
        # motion_data = motion_data[None, :]
        
        # window_size = 512
        # midi_samples = []
        # motion_samples = []
        # print(midi_data.shape)
        
        # for midi_sample, motion_sample in zip(midi_data, motion_data):
        #     print("midi_sample", midi_sample.shape)
        #     print("midi_sample.size(0)", midi_sample.shape[0])
        #     midi_sample_len = midi_sample.shape[0]-1
        #     for i in range(0, midi_sample.shape[0], window_size): #or window_size/2
        #         segment_midi = np.array(midi_sample[i:i+window_size])
        #         segment_motion = np.array(motion_sample[i:i+window_size])
                
        #         segment_midi_len = len(segment_midi)
        #         print("before: ", len(segment_midi))
        #         if segment_midi_len < window_size:
        #             segment_midi = midi_sample[(midi_sample_len-window_size):(midi_sample_len-window_size) + window_size]
        #             segment_motion = motion_sample[(midi_sample_len-window_size):(midi_sample_len-window_size) + window_size]
        #             print("after: ", len(segment_midi))
        #         midi_samples.append(segment_midi)
        #         motion_samples.append(segment_motion)
        #         print("segment_midi.shape:", segment_midi.shape)
        #         print("segment_motion.shape:", segment_motion.shape)

        # print(len(segment_midi), len(segment_motion))
        return segment_midi, segment_motion

        # , self.music_list[index]
        # print(midi_data.shape)
        # print(motion_data.shape)
        # return midi_data, motion_data  # self.performer_namecodes[index]


# class MidiMotionCollate():
#     def __init__(self, device):
#         super().__init__()
#         self.device = device

#     def __call__(self, batch):
#         B = len(batch)
        
#         # midi_dim = batch[0][0][0].shape[1]  # 128
#         # motion_dim = batch[0][1][0].shape[1]  # 102
#         # print("midi_dim: ", midi_dim)
#         # print("motion_dim: ", motion_dim)
#         # # have same length
#         # midi_len = [data[0][0].shape[0] for data in batch]
#         # motion_len = [data[0][1].shape[0] for data in batch]

#         # max_len = max(midi_len)
#         # print("max_len:", max_len)

#         pad_midi = torch.zeros((B, max_len, midi_dim), dtype=torch.int32)
#         pad_motion = torch.zeros(
#             (B, max_len, motion_dim), dtype=torch.float)

#         # # pad_mask_midi = torch.arange(midi_len)
#         # # pad_mask_motion = torch.arange(motion_len)

#         for i, (midi_data, motion_data) in enumerate(batch):
#             pad_midi[j, :midi_data[0].shape[0], :] = torch.Tensor(midi_data[0])
#             pad_motion[j, :motion_data[0].shape[0], :] = torch.Tensor(motion_data[0])

#         # torch.Tensor(midi_data).unsqueeze(0)
#         # torch.Tensor(motion_data).unsqueeze(0)
#         print(pad_midi.shape)
#         print(pad_motion.shape)

#         pad_midi = pad_midi.to(self.device)
#         pad_motion = pad_motion.to(self.device)
#         # pad_mask_midi = pad_mask_midi.to(self.device)
#         # pad_mask_motion = pad_mask_motion.to(self.device)

#         return pad_midi, pad_motion
        # return pad_midi, pad_motion, stop_token, pad_mask_midi, pad_mask_motion


def get_dataloader(dataset_path, batch_size=1, device='cuda'):
    dataset = MidiMotionDataSet(dataset_path)
    # music_list = dataset.get_music_list()
    # collate_feature = MidiMotionCollate(device)
    data_loader = DataLoader(dataset,
                             num_workers=0,
                             shuffle=True,
                             sampler=None,
                             batch_size=batch_size,
                             pin_memory=False,
                             drop_last=False)#collate_fn=collate_feature
    return data_loader  # , music_list


if __name__ == "__main__":
    dataset_name_path = f"./midi_list.txt"
    dataloader = get_dataloader(dataset_name_path, batch_size=20)

    for pad_midi, pad_motion in dataloader:
        print(pad_midi.shape, pad_motion.shape)
