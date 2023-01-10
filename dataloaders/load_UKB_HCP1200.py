import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import mat73
import scipy.io
import time

class UKBHCP1200Data(Dataset):
    def __init__(self,
                 N_components=53,
                 N_timepoints=448,
                #  window_size=40,
                 N_subs=200,
                 #data_root="/data/qneuromark/Results/DFNC/UKBioBank",
                 #subj_form="%s", #"Sub%05d",
                 #data_file="UKB_dfnc_sub_001_sess_001_results.mat",
                 age_csv="/data/users2/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths.csv",
                #  age_threshold=15
                converted_csv="./data/converted_files.csv"
                 ):
        """DevData - Load FNCs for UKBiobank
        KWARGS:
            N_components    int     the number of ICs
            N_timepoints    int     the length of the scan
            window_size     int     window size for dFNC
            N_subs          int     number of subjects
            data_root       str     root directory for loading subject data
            subj_form       strf    format string for resolving subject folders
            data_file       str     the filename for loading individual subject data
        """
        self.N_subjects = N_subs
        # The dFNC filepath for each subject is a format string, where the root and filename stay the same
        # but subject directory changes
        #self.filepath_form = os.path.join(data_root, subj_form, data_file)
        
        # Compute the size of the upper-triangle in the FNC matrix
        upper_triangle_size = int(N_components*(N_components - 1)/2)
        # ts_length = N_timepoints - window_size
        # These variables are useful for properly defining the RNN used
        self.dim = upper_triangle_size
        self.seqlen = N_timepoints
        # Load demographic data (small enough to keep in memory)
        # UKB_demo = pd.read_csv(age_csv)
        # UKB_demo_clean = UKB_demo[UKB_demo.age > age_threshold]
        all_data = pd.read_csv(age_csv)
        np.random.seed(319)
        self.idxs = np.random.permutation(len(all_data))[:N_subs]
        self.age = np.array(all_data.loc[self.idxs, "age"]).reshape(
            N_subs, 1).astype('float32')
        self.subID = np.array(all_data.loc[self.idxs, "Subject"])
        self.filepath = np.array(all_data.loc[self.idxs, "DFNC_full_fname"])
        self.converted = None
        if converted_csv:
            self.converted = pd.read_csv(converted_csv)

    def __len__(self):
        """Returns the length of the dataset
            i.e. the Number of subjects
        """
        return self.N_subjects

    def __getitem__(self, k):
        """Get an individual FNC matrix (flattened) and Age (integer), resolving filepath format string with index
            and using mat73 to load data
        """        
        filepath = self.filepath[k]
        if self.converted is not None and filepath == self.converted['old_filename'].iloc[k]:
            filepath = self.converted['new_filename'].iloc[k]
            data = torch.load(filepath)
            dfnc = data[:self.seqlen,:].float()
            return dfnc, torch.from_numpy(self.age[k])
        else:
            try:
                data = mat73.loadmat(filepath)
            except:
                data = scipy.io.loadmat(filepath)
            dfnc = data['FNCdyn'][:self.seqlen,:].astype('float32')
            return torch.from_numpy(dfnc), torch.from_numpy(self.age[k])            
        


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    test_dataset = DevData(N_subs=200)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    for batch_i, (fnc, age) in enumerate(test_dataloader):
        print("Loaded batch %d with FNC shape %s, and average age %.2f" %
              (batch_i, str(fnc.shape), age.mean().item()))
