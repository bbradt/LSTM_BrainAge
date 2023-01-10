import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import mat73
import scipy.io
import time


class CadasilData(Dataset):
    def __init__(self,
                 N_components=53,
                 N_timepoints=929,
                 N_subs=15,
                 phenotypes="/data/users2/bbaker/projects/cadasil_analysis/CADASIL_phenotype_full.csv",
                 target_variable="age"
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
        all_data = pd.read_csv(phenotypes)
        self.idxs = list(range(N_subs))
        self.target = np.array(all_data.loc[self.idxs, target_variable]).reshape(
            N_subs, 1).astype('float32')
        self.subID = np.array(all_data.loc[self.idxs, "URSI"])
        self.filepath = np.array(all_data.loc[self.idxs, "filename"])

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
        data = torch.load(filepath)
        dfnc = data[:self.seqlen, :].astype("float32")
        return dfnc, torch.from_numpy(self.target[k])


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    test_dataset = CadasilData(N_subs=22)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    for batch_i, (fnc, age) in enumerate(test_dataloader):
        print("Loaded batch %d with FNC shape %s, and average age %.2f" %
              (batch_i, str(fnc.shape), age.mean().item()))
