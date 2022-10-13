import os
import io

import sys
import csv

import json
import glob
import tqdm
import subprocess

import pandas as pd
import matplotlib.pyplot as plt

from pydub import AudioSegment
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

from bvh_converter.bvhplayer_skeleton import process_bvhfile, process_bvhkeyframe

ffmpeg_exe_path = r'C:\FFmpeg\bin\ffmpeg.exe'

class BVHReader:
    """
    BVHReader is used to extract 3D position data from mocap .bvh file.
    --------
    Args:
    --------
    save_folder: the target folder for save output csv files.
    ----------------
    Methods:
    --------
    open_csv: help to read csv file.
    proc_one_file: process a single bvh file.
    proc_folder: process all bvh files in a folder.
    """
    def __init__(self, save_folder):
        self.save_folder = save_folder

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)  

    def open_csv(self, filename, mode='r'):
        """Open a csv file in proper mode depending on Python version."""
        if sys.version_info < (3,):
            return io.open(filename, mode=mode+'b')
        else:
            return io.open(filename, mode=mode, newline='')

    def proc_one_file(self, file_in, need_rot=False):

        if not os.path.exists(file_in):
            print("Error: file {} not found.".format(file_in))
            sys.exit(0)

        other_s = process_bvhfile(file_in)

        for i in range(other_s.frames):
            new_frame = process_bvhkeyframe(other_s.keyframes[i], other_s.root,
                                            other_s.dt * i)
        
        file_out = file_in[:-4] + "_worldpos.csv"
        file_rot_out = ''

        with self.open_csv(self.save_folder+'/'+file_out.split('\\')[-1], 'w') as f:
            writer = csv.writer(f)
            header, frames = other_s.get_frames_worldpos()
            writer.writerow(header)
            for frame in frames:
                writer.writerow(frame)   

        if need_rot:
            file_rot_out = file_in[:-4] + "_rotations.csv"
        
            with self.open_csv(self.save_folder+'/'+file_rot_out.split('\\')[-1], 'w') as f:
                writer = csv.writer(f)
                header, frames = other_s.get_frames_rotations()
                writer.writerow(header)
                for frame in frames:
                    writer.writerow(frame)

    def proc_folder(self, tgt_folder, need_rot=False):

        for file_in in tqdm.tqdm(glob.glob(tgt_folder+'/*.bvh')):

            self.proc_one_file(file_in=file_in, need_rot=need_rot)

        print('Done!')

class KinectProcessor:
    """
    FacePt is used to process kinect-captured face points, especially, lip points.
    --------
    Args:
    --------
    root_dir: str, the directory saved kinect data.
    save_dir: str, the directory used to save extracted frame files.
    """
    def __init__(self, root_dir, save_dir):
        self.lip_pts_name = [
            'FaceJoint16',
            'FaceJoint27End', 'FaceJoint26End', 'FaceJoint25End', 
            'FaceJoint24End', 'FaceJoint23End', 'FaceJoint22End', 
            'FaceJoint21End','FaceJoint20End', 'FaceJoint19End', 
            'FaceJoint18End', 'FaceJoint17End','FaceJoint16End'
            ]
        
        self.face_pts_name = ['FaceJoint28'] + ['FaceJoint{}End'.format(id) for id in range(0, 29)]

        self.root_dir = root_dir
        self.save_dir = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)    

        if not os.path.exists(save_dir + '/videos'):
            os.makedirs(save_dir + '/videos')
            
        if not os.path.exists(save_dir + '/audios'):
            os.makedirs(save_dir + '/audios')

        if not os.path.exists(save_dir + '/movies'):
            os.makedirs(save_dir + '/movies')
            
        if not os.path.exists(save_dir + '/landmarkers'):
            os.makedirs(save_dir + '/landmarkers')

    def extract_frames_folder(self):
        """
        Given the root directory, extract frames of all files according to the start and end time
        and save extracted frames into a save_folder.
        """

        for file in tqdm.tqdm(glob.glob(self.root_dir+'/landmarkers/*.csv')):
            df = pd.read_csv(file)

            temp = file.split('\\')[-1].replace('.csv', '.json')
            temp = temp.replace('land', 'timestamp')

            timestamp_file = self.root_dir + '/timestamps/' + temp

            # extract landmakers csv file
            df_temp = self._extract_frames_onefile(df, timestamp_file)
            df_temp.to_csv(self.save_dir+'/landmarkers/'+file.split('\\')[-1].replace('land', 'land_proc'))

            # extract audio wav file
            temp = file.split('\\')[-1].replace('.csv', '.wav')
            temp = temp.replace('land', 'audio')
            end_time = (df_temp.Time[len(df_temp)-1] - df_temp.Time[0]) * 1000 # works in milliseconds
            new_audio = AudioSegment.from_wav(self.root_dir + '/audios/' + temp)
            new_audio = new_audio[:end_time]
            new_audio.export(self.save_dir+'/audios/'+temp.replace('audio', 'audio_proc'), format='wav')
            audio_name = self.save_dir+'/audios/'+temp.replace('audio', 'audio_proc')

            # extract video avi file
            temp = file.split('\\')[-1].replace('.csv', '.avi')
            temp = temp.replace('land', 'video')

            # loading video 
            ffmpeg_extract_subclip(self.root_dir+'/videos/'+temp, 0, df_temp.Time[len(df_temp)-1] - df_temp.Time[0], 
                          targetname=self.save_dir+'/videos/'+temp.replace('video', 'video_proc'))
            video_name = self.save_dir+'/videos/'+temp.replace('video', 'video_proc')

            # combine video and audio
            command = ffmpeg_exe_path + ' -i ' + video_name + ' -i ' + audio_name + ' -c:v copy -c:a aac ' + self.save_dir + '/movies/' + temp.replace('video', 'movie')
            
            subprocess.run(command.replace('\\', '/'))


    def _extract_frames_onefile(self, df, timestamp_file):
        """
        Given the start time and end time, extract frames within the duration.
        --------
        Args:
        --------
        df: pandas.DataFrame, contains time step and 3d position information. The key of time step column should be 'Time'.
        timestamp_file: str, json file store start and end timestamps.
        """

        with open(timestamp_file, 'r') as f:
            times = json.load(f)

        timestamps = self._calcu_ts_onefile(df, times)

        for i, ts in enumerate(timestamps):
            if ts > times['end_time']:
                break

        final_df = df.iloc[0:i].copy()
        final_df.Time = timestamps[0:i]

        return final_df

    def _calcu_ts_onefile(self, df, times):
        """
        Given the start time, calculate the timestamps for every frames.
        --------
        Args:
        --------
        df: pandas.DataFrame, contains time step and 3d position information. The key of time step column should be 'Time'.
        times: dict, have keys: start_time, end_time, start_dtime, end_dtime.
        """

        timestamps = []

        start_time = times['start_time']

        for i in range(len(df)):

            timestamps.append(start_time+df.Time[i])

        return timestamps

    def ani_motion_onefile(self, file, mode='lip', pause=0.01):
        """
        Given a mocap data, animate movements of recorded points in 3d axes.
        --------
        Args:
        --------
        df: pandas.DataFrame, contains time step and 3d position information. The key of 3d position columns should be one of lip_pts_name/face_pts_name + '.X/Y/Z'.
        mode: str, availale options are: face and lip.
        pause: float, control animation play speed.
        """
        
        df = pd.read_csv(file)

        show_pts = []
        
        if mode == 'lip':
            show_pts = self.lip_pts_name.copy()
        elif mode == 'face':
            show_pts = self.face_pts_name.copy()
        else:
            raise Exception('Wrong mode: available modes 1. face, 2. lip')

        plt.ion() # enable interactive control
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for f in range(len(df)):
            ax.cla()

            x = [df[pt+'.X'][f] for pt in show_pts]
            y = [df[pt+'.Y'][f] for pt in show_pts]
            z = [df[pt+'.Z'][f] for pt in show_pts]
            ax.scatter(x, y, z)
            
            plt.draw()
            plt.pause(pause)
            
        