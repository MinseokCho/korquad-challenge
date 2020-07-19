
import torch
import sys

from torch.utils.data import Dataset

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class KorquadDataset(Dataset):

    def __init__(self, pkl_list, batch_size, tokenizer):

        self.pkl_idx = 0
        self.pkl_list = pkl_list
        self.batch_size = batch_size
        self.tokenizer = tokenizer

        self.feature_idx = 0
        self.features = self.get_feature()
        self.num_features = self.features["data_num"]

    def next_batch(self):

        idx = slice(self.feature_idx, self.feature_idx+self.batch_size, None)

        if (idx.stop > self.num_features) and (self.pkl_idx != len(self.pkl_list)-1):
            idx = slice(self.feature_idx, self.num_features, None)

            self.pkl_idx += 1
            new_feature = self.get_feature()
            
            self.feature_idx = self.batch_size - (idx.stop-idx.start)
            new_idx = slice(0, self.feature_idx, None)
            result = tuple([torch.cat((self.features["all_input_ids"][idx], new_feature["all_input_ids"][new_idx]), 0), 
                torch.cat((self.features["all_input_mask"][idx], new_feature["all_input_mask"][new_idx]), 0), 
                torch.cat((self.features["all_segment_ids"][idx], new_feature["all_segment_ids"][new_idx]), 0), 
                torch.cat((self.features["all_start_positions"][idx], new_feature["all_start_positions"][new_idx]), 0), 
                torch.cat((self.features["all_end_positions"][idx], new_feature["all_end_positions"][new_idx]), 0)])

            del self.features
            self.features = new_feature
            self.num_features = self.features["data_num"]

        else:
            self.feature_idx = idx.stop
            result = tuple([self.features["all_input_ids"][idx], self.features["all_input_mask"][idx], self.features["all_segment_ids"][idx], self.features["all_start_positions"][idx], self.features["all_end_positions"][idx]])
            if (self.pkl_idx == len(self.pkl_list)-1) and (self.feature_idx >= self.num_features):
                self.pkl_idx = 0
                
                del self.features
                self.feature_idx = 0
                self.features = self.get_feature()
                self.num_features = self.features["data_num"]

        return result


    def get_feature(self):
        with open(self.pkl_list[self.pkl_idx], 'rb') as f:
            return pickle.load(f)
