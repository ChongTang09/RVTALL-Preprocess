import os

import glob
import tqdm

import numpy as np

from scipy.io import loadmat

from basic_proc import BasicProc

class UWBProcessor(BasicProc):
    """
    UWBProcessor is used to segment and preprocess UWB spectrogram
    --------
    Args:
    --------
    root_dir: str, the directory saved UWB data.
    """
    def __init__(self, root_dir):
        
        super(UWBProcessor, self).__init__()
        
        self.lookup_dict = {'1': 'vowel', '2': 'word', '3': 'sentences'}

        self.root_dir = root_dir.replace('\\', '/')
        
    def _segment_one_exp(self, uwbmat_file):
        """
        Parameters
        ----------
        uwbmat_file : str
            file name of UWB mat: EXPID_TASKID_PERSONID_RADARID_xethru.mat.

        Returns
        -------
        None.
        """
        exp_info = uwbmat_file.split('_')
        
        uwbmat = loadmat(self.root_dir + '/UWB_Person_' + exp_info[2] + '/' + uwbmat_file)
        timestamp_folder = self.root_dir + '/Kinect_Person_' + exp_info[2] + '/' + self.lookup_dict[exp_info[1]] + exp_info[0] + '/timestamps'
        save_dir = self.root_dir + '/UWB_Person_' + exp_info[2] + '/' + self.lookup_dict[exp_info[1]] + exp_info[0]
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  
        
        cnt = 1
        for timestamp_file in tqdm.tqdm(glob.glob(timestamp_folder + '/*.json')):
            try:
                uwb_sample = self._segment_one_sample(uwbmat, timestamp_file)
                np.save(save_dir + '/sample' + str(cnt) + '.npy', uwb_sample)
                cnt += 1
            except:
                cnt += 1
                continue
            
        
    def _segment_one_sample(self, uwbmat, timestamp_file):
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
        frames_time = self._calcu_timestamp_uwbframes(uwbmat)
        
        kinec_start_time = self._datetime2unixtimestamp(self._iosstr2datetime(times['start_dtime'], "%Y-%m-%d %H:%M:%S.%f"))
        kinec_end_time = self._datetime2unixtimestamp(self._iosstr2datetime(times['end_dtime'], "%Y-%m-%d %H:%M:%S.%f"))
        
        start_idx, end_idx = self._match_start_end_ts(frames_time, kinec_start_time, kinec_end_time)
        
        uwb_sample = uwbmat['Data_MicroDop_2'][:, start_idx:end_idx+1]
        
        return uwb_sample
        
    def _calcu_timestamp_uwbframes(self, uwbmatobj):
        """
        Parameters
        ----------
        uwbmatobj : mat object
            contains UWB spectrogram and time information.

        Returns
        -------
        frames_time : list
            timestamp for each frame of UWB mat.
        """
        frames_time = [] 
        
        frames_no = uwbmatobj['Data_MicroDop_2'].shape[1]
        start_dtime_str = uwbmatobj['Starttime'][0]
        end_dtime_str = uwbmatobj['Endtime'][0]
        
        start_time = self._datetime2unixtimestamp(self._iosstr2datetime(start_dtime_str))
        end_time = self._datetime2unixtimestamp(self._iosstr2datetime(end_dtime_str))
        time_interval = end_time - start_time
        
        for i in range(frames_no):

            frames_time.append(start_time+i*time_interval/(frames_no-1))

        return frames_time