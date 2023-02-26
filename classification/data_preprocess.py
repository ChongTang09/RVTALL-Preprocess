import tqdm
import torch
import glob

import numpy as np
from matplotlib import image
from torchvision import transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def mat2img(npy_path):

    for npy in glob.glob(npy_path+'/*.npy'):
        arr = np.load(npy)
        im = image.imsave(npy.replace('npy', 'png'), 20*np.log10(abs(arr)), cmap='jet')

def load_data(task_path, mode):
    samples = []

    if mode == 'kinect':
        for video in glob.glob(task_path+'/videos/*.avi'):
            sample = []
            img_path = video.replace('.avi', '')
            for im_dir in glob.glob(img_path+'/mouths/*.png'):
                im = Image.open(im_dir).convert('L')
                sample.append(im)

            samples.append(sample)
    elif mode == 'radar':
        for im_dir in glob.glob(task_path+'/*.png'):
            im = Image.open(im_dir).convert('L')
            samples.append(im)
    elif mode == 'uwb':
        for im_dir in glob.glob(task_path+'/*.png'):
            im = Image.open(im_dir).convert('L')
            samples.append(im)
    else:
        raise Exception('Invalid mode.')
    
    return samples



# if __name__ == '__main__':
#     root_radar = r'D:\Glasgow\RVTALL\processed_cut_data\radar_processed'
#     root_uwb = r'D:\Glasgow\RVTALL\processed_cut_data\uwb_processed'
#     subjects = [str(i) for i in range(1, 21)]
#     sentences = ['sentences_'+str(i) for i in range(1, 11)]
#     words = ['word_'+str(i) for i in range(1, 16)]
#     vowels = ['vowel_'+str(i) for i in range(1, 6)]

#     for sub in tqdm.tqdm(subjects):
#         for sent in sentences:
#             # mat2img(npy_path=root_radar+'/'+sub+'/'+sent)
#             mat2img(npy_path=root_uwb+'/'+sub+'/'+sent)
#         for word in words:
#             # mat2img(npy_path=root_radar+'/'+sub+'/'+word)
#             mat2img(npy_path=root_uwb+'/'+sub+'/'+word)
#         for vowel in vowels:
#             # mat2img(npy_path=root_radar+'/'+sub+'/'+vowel)
#             mat2img(npy_path=root_uwb+'/'+sub+'/'+vowel)