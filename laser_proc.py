import os

import tqdm
import glob

import numpy as np

from scipy.io import loadmat

from basic_proc import BasicProc

class LaserProcessor(BasicProc):
    
    def __init__(self, root_dir):
        super(LaserProcessor, self).__init__()
        
        self.root_dir = root_dir.replace('\\', '/')
        
    def lookup_dict(self, index):
        """
        Parameters
        ----------
        index : int
            the index of task

        Returns
        -------
        task name, real index
        """
        if index >= 1 and index <=5:
            return 'vowel', index
        elif index >= 6 and index <= 20:
            return 'word', int(index-5)
        elif index >= 21 and index <= 30:
            return 'sentences', int(index-20)
        
    def _segment_one_exp(self, lasermat_file, index):
        """
        Parameters
        ----------
        uwbmat_file : str
            file name of UWB mat: EXPID_TASKID_PERSONID_RADARID_xethru.mat.

        Returns
        -------
        None.
        """
        exp_info = lasermat_file.replace('.mat', '').split('_')
        
        lasermat = loadmat(self.root_dir + '/Laser/' + lasermat_file)
        timestamp_folder = self.root_dir + '/Kinect_Person_' + exp_info[1] + '/' + self.lookup_dict(index)[0] + str(self.lookup_dict(index)[1]) + '/timestamps'
        save_dir = self.root_dir + '/Laser/' + self.lookup_dict(index)[0] + str(self.lookup_dict(index)[1])
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  
        
        cnt = 1
        for timestamp_file in tqdm.tqdm(glob.glob(timestamp_folder + '/*.json')):
            try:
                laser_sample = self._segment_one_sample(lasermat, timestamp_file, index)
                np.save(save_dir + '/sample' + str(cnt) + '.npy', laser_sample)
                cnt += 1
            except:
                cnt += 1
                continue
            
    def _segment_one_sample(self, lasermatobj, timestamp_file, index):
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
        frames_time = self._calcu_timestamp_mmWaveframes(lasermatobj, index)
        
        kinec_start_time = self._datetime2unixtimestamp(self._iosstr2datetime(times['start_dtime'], "%Y-%m-%d %H:%M:%S.%f"))
        kinec_end_time = self._datetime2unixtimestamp(self._iosstr2datetime(times['end_dtime'], "%Y-%m-%d %H:%M:%S.%f"))
        
        start_idx, end_idx = self._match_start_end_ts(frames_time, kinec_start_time, kinec_end_time)
        
        laser_sample = lasermatobj['datatimestep_02'][int(index-1)][start_idx:end_idx+1]
        
        return laser_sample
        
    def _calcu_timestamp_mmWaveframes(self, lasermatobj, index):
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
        frames_time = [] 
        
        frames_no = lasermatobj['datatimestep_02'].shape[1]
        start_dtime_str = '-'.join([str(int(x)) for x in lasermatobj['datatimestart_02'][int(index-1)][0:-1]] + [str(lasermatobj['datatimestart_02'][int(index-1)][-1])])
        end_dtime_str = '-'.join([str(int(x)) for x in lasermatobj['datatimestop_02'][int(index-1)][0:-1]] + [str(lasermatobj['datatimestop_02'][int(index-1)][-1])])
        
        start_time = self._datetime2unixtimestamp(self._iosstr2datetime(start_dtime_str, form="%Y-%m-%d-%H-%M-%S.%f"))
        end_time = self._datetime2unixtimestamp(self._iosstr2datetime(end_dtime_str, form="%Y-%m-%d-%H-%M-%S.%f"))
        time_interval = end_time - start_time
        
        for i in range(frames_no):

            frames_time.append(start_time+i*time_interval/(frames_no-1))

        return frames_time