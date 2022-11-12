import os

import tqdm
import glob

import numpy as np

from scipy.io import loadmat

from basic_proc import BasicProc

class mmWaveProcessor(BasicProc):
    
    def __init__(self, root_dir):
        super(mmWaveProcessor, self).__init__()
        
        self.lookup_dict = {'1': 'vowel', '2': 'word', '3': 'sentences'}

        self.root_dir = root_dir.replace('\\', '/')
        
    def _segment_one_exp(self, mmwmat_file):
        """
        Parameters
        ----------
        uwbmat_file : str
            file name of UWB mat: EXPID_TASKID_PERSONID_RADARID_xethru.mat.

        Returns
        -------
        None.
        """
        exp_info = mmwmat_file.replace('.mat', '').split('_')
        
        mmwmat = loadmat(self.root_dir + '/mmWave_Person_' + exp_info[0] + '/' + mmwmat_file)
        timestamp_folder = self.root_dir + '/Kinect_Person_' + exp_info[0] + '/' + self.lookup_dict[exp_info[1]] + exp_info[2] + '/timestamps'
        save_dir = self.root_dir + '/mmWave_Person_' + exp_info[0] + '/' + self.lookup_dict[exp_info[1]] + exp_info[2]
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  
        
        cnt = 1
        for timestamp_file in tqdm.tqdm(glob.glob(timestamp_folder + '/*.json')):
            try:
                mmw_sample = self._segment_one_sample(mmwmat, timestamp_file)
                np.save(save_dir + '/sample' + str(cnt) + '.npy', mmw_sample)
                cnt += 1
            except:
                cnt += 1
                continue
            
    def _segment_one_sample(self, mmwmat, timestamp_file):
        """
        Parameters
        ----------
        matfile : str
            location of UWB mat file.
        timestamp_file : str
            location of timestamp file.

        Returns
        -------
        uwb_sample : np.array
            a uwb spectrogram segment.
        """
        times = self._loadtimestamp(timestamp_file)
        frames_time = self._calcu_timestamp_mmWaveframes(mmwmat, timestamp_file)
        
        kinec_start_time = self._datetime2unixtimestamp(self._iosstr2datetime(times['start_dtime'], "%Y-%m-%d %H:%M:%S.%f"))
        kinec_end_time = self._datetime2unixtimestamp(self._iosstr2datetime(times['end_dtime'], "%Y-%m-%d %H:%M:%S.%f"))
        
        start_idx, end_idx = self._match_start_end_ts(frames_time, kinec_start_time, kinec_end_time)
        
        mmw_sample = mmwmat['s'][:, start_idx:end_idx+1]
        
        return mmw_sample
        
    def _calcu_timestamp_mmWaveframes(self, mmwmatobj, timestamp_file):
        """
        Parameters
        ----------
        uwbmatobj : mat object
            contains mmWave spectrogram and time information.

        Returns
        -------
        frames_time : list
            timestamp for each frame of UWB mat.
        """
        times = self._loadtimestamp(timestamp_file)
        
        frames_time = [] 
        
        frames_no = mmwmatobj['s'].shape[1]
        start_dtime_str = times['start_dtime'][0:11] + mmwmatobj['TiTime'][0][0][0][0][0][1:13]
        end_dtime_str = times['start_dtime'][0:11] + mmwmatobj['TiTime'][0][0][2][0][0][1:13]
        
        start_time = self._datetime2unixtimestamp(self._iosstr2datetime(start_dtime_str, form="%Y-%m-%d %H:%M:%S:%f"))
        end_time = self._datetime2unixtimestamp(self._iosstr2datetime(end_dtime_str, form="%Y-%m-%d %H:%M:%S:%f"))
        time_interval = end_time - start_time
        
        for i in range(frames_no):

            frames_time.append(start_time+i*time_interval/(frames_no-1))

        return frames_time