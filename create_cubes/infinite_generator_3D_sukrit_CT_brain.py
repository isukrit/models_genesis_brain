#!/usr/bin/env python
# coding: utf-8

"""
for subset in `seq 1 9`
do
python -W ignore infinite_generator_3D_sukrit_CT_brain.py \
--fold $subset \
--scale 32 \
--data ../CQ500_data \
--save generated_cubes_CT
done
"""

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import os
#import keras
#print("Keras = {}".format(keras.__version__))
#import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import string
import sys
import math
import random
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import nibabel as nib

from tqdm import tqdm
from sklearn import metrics
from optparse import OptionParser
from glob import glob
from skimage.transform import resize
import pydicom
import imageio
from skimage import measure, morphology
from mpl_toolkits.mplot3d import axes3d, Axes3D

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--fold", dest="fold", help="fold of subset", default=None, type="int")
parser.add_option("--input_rows", dest="input_rows", help="input rows", default=64, type="int")
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=64, type="int")
parser.add_option("--input_deps", dest="input_deps", help="input deps", default=32, type="int")
parser.add_option("--crop_rows", dest="crop_rows", help="crop rows", default=64, type="int")
parser.add_option("--crop_cols", dest="crop_cols", help="crop cols", default=64, type="int")
parser.add_option("--data", dest="data", help="the directory of LUNA16 dataset", default=None, type="string")
parser.add_option("--save", dest="save", help="the directory of processed 3D cubes", default=None, type="string")
parser.add_option("--scale", dest="scale", help="scale of the generator", default=32, type="int")
(options, args) = parser.parse_args()
fold = options.fold

seed = 1
random.seed(seed)

assert options.data is not None
assert options.save is not None
assert options.fold >= 0 and options.fold <= 10

if not os.path.exists(options.save):
    os.makedirs(options.save)

def plot_3d(image, save_img_dir, threshold):

  # Position the scan upright, 
  # so the head of the patient would be at the top facing the camera
  p = image.astype(np.uint8) #.transpose(2,1,0)

  try:
    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    #ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.savefig(save_img_dir)
    return
  except:
    return
  
class setup_config():

    hu_max = 1000.0
    hu_min = -1000.0

    ''' MRI
    hu_max = 100
    hu_min = 0
    '''

    HU_thred = (40 - hu_min) / (hu_max - hu_min)
    def __init__(self, 
                 input_rows=None, 
                 input_cols=None,
                 input_deps=None,
                 crop_rows=None, 
                 crop_cols=None,
                 len_border=None,
                 len_border_z=None,
                 scale=None,
                 DATA_DIR=None,
                 train_fold=[0,1,2,3,4],
                 valid_fold=[5,6],
                 test_fold=[7,8,9],
                 len_depth=None,
                 lung_min=0.7,
                 lung_max=1.0,
                 save_samples = True
                ):
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.input_deps = input_deps
        self.crop_rows = crop_rows
        self.crop_cols = crop_cols
        self.len_border = len_border
        self.len_border_z = len_border_z
        self.scale = scale
        self.DATA_DIR = DATA_DIR
        self.train_fold = train_fold
        self.valid_fold = valid_fold
        self.test_fold = test_fold
        self.len_depth = len_depth
        self.lung_min = lung_min
        self.lung_max = lung_max
        self.save_samples = save_samples

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")



config = setup_config(input_rows=options.input_rows,
                      input_cols=options.input_cols,
                      input_deps=options.input_deps,
                      crop_rows=options.crop_rows,
                      crop_cols=options.crop_cols,
                      scale=options.scale,
                      len_border=3,
                      len_border_z=0,
                      len_depth=3,
                      lung_min=0.1,
                      lung_max=1.0,
                      DATA_DIR=options.data,
                      save_samples = False
                     )
config.display()

def infinite_generator_from_one_volume(config, img_array):
    size_x, size_y, size_z = img_array.shape
    if size_z-config.input_deps-config.len_depth-1-config.len_border_z < config.len_border_z:
        return None
    
    img_array[img_array < config.hu_min] = config.hu_min
    img_array[img_array > config.hu_max] = config.hu_max
    img_array = 1.0*(img_array-config.hu_min) / (config.hu_max-config.hu_min)
    img_array = img_array

    print ('Max in an image: ', np.max(img_array), 'Min in the image: ', np.min(img_array))

    slice_set = np.zeros((config.scale, config.input_rows, config.input_cols, config.input_deps), dtype=float)
    
    num_pair = 0
    cnt = 0
    while True:
        cnt += 1
        if cnt > 50 * config.scale and num_pair == 0:
            return None
        elif cnt > 50 * config.scale and num_pair > 0:
            return np.array(slice_set[:num_pair])
        
        start_x = random.randint(0+config.len_border, size_x-config.crop_rows-1-config.len_border)
        start_y = random.randint(0+config.len_border, size_y-config.crop_cols-1-config.len_border)
        start_z = random.randint(0+config.len_border_z, size_z-config.input_deps-config.len_depth-1-config.len_border_z)
        
        crop_window = img_array[start_x : start_x+config.crop_rows,
                                start_y : start_y+config.crop_cols,
                                start_z : start_z+config.input_deps+config.len_depth]


        if config.crop_rows != config.input_rows or config.crop_cols != config.input_cols:
            crop_window = resize(crop_window, 
                                 (config.input_rows, config.input_cols, config.input_deps+config.len_depth), 
                                 preserve_range=True)
        
        t_img = np.zeros((config.input_rows, config.input_cols, config.input_deps), dtype=float)
        d_img = np.zeros((config.input_rows, config.input_cols, config.input_deps), dtype=float)

        for d in range(config.input_deps):
            for i in range(config.input_rows):
                for j in range(config.input_cols):
                    for k in range(config.len_depth): #check in a depth of d+k around the 2D image if any pixel exceeds the threshold. If it doesn't (break statement not executed), then put the values from slice (d+k-1) into the pixel.  
                        if crop_window[i, j, d+k] >= config.HU_thred:
                            t_img[i, j, d] = crop_window[i, j, d+k]
                            d_img[i, j, d] = k #stores the depth from which the (i,j)th pixel in slice d has been taken.
                            break
                        if k == config.len_depth-1:
                            d_img[i, j, d] = k
                            
        d_img = d_img.astype('float32') #pixels taken from their own slice will have d_img[i, j, d] = 0. 
        d_img /= (config.len_depth - 1)
        d_img = 1.0 - d_img
        
        cube_vols = config.input_rows * config.input_cols * config.input_deps

        if np.sum(d_img) > config.lung_max * cube_vols: #just ensures that not too many pixels are taken from slices far away than their original slice.
            print (np.sum(d_img), config.lung_max * config.input_rows * config.input_cols * config.input_deps)
            continue

        if np.sum(t_img) < config.lung_min * cube_vols or np.sum(t_img) > (1- config.lung_min) * cube_vols: #check if there is enough information in the cube.
            print ('Not enough information in cube ', num_pair, 'max:', cube_vols, 'information: ', np.sum(t_img))
            continue

        slice_set[num_pair] = crop_window[:,:,:config.input_deps]
        
        print ('slice_set[num_pair]', slice_set[num_pair].shape)

        if config.save_samples:

            final_sample = slice_set[num_pair] #* 255
            final_sample = final_sample.astype(np.uint8)

            file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)]) + '.png'
            image_path = os.path.join(options.save, 'CT_cube_samples')
            
            if not os.path.exists(image_path):
                os.makedirs(image_path) 
            
            plot_3d(final_sample, os.path.join(image_path, file_name), threshold = np.min(final_sample))


            
            #imageio.imwrite(, final_sample)

            print ("************ Wrote Images *****************")

        num_pair += 1
        if num_pair == config.scale:
            break

    return np.array(slice_set)

def load_scan(path, subject_id, session):
  print ('Subject:', subject_id, 'Session', session)
  slices = []
  try:
    for s in os.listdir(path): 
      slices.append(pydicom.dcmread(path + '/' + s, force = True))
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
  except:
    return None

  try:
    slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
  except:
    pass

  try:
    slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
  except:
    return None   
  
  for s in slices:
    s.SliceThickness = slice_thickness

  img_shape = list(slices[0].pixel_array.shape)
  img_shape.append(len(slices))
  img3d = np.zeros(img_shape)

  # fill 3D array with the images from the files
  for i, s in enumerate(slices):
    img2d = s.pixel_array
    img3d[:, :, i] = img2d
      
  return img3d

def get_self_learning_data(fold, config):
    slice_set = []
    labels_set = []
    sites = fold
    subject_label_data = np.genfromtxt(os.path.join(config.DATA_DIR, 'reads.csv'), delimiter=',', dtype='unicode')


    subject_label_mapping = dict()

    for row in subject_label_data[1:]:    #B1: CN, B2: Stroke
      sub_id = row[0].replace('-', '')
      if row[-1] == '0':
        subject_label_mapping[sub_id] = 0
      if row[-1] != '0':
        subject_label_mapping[sub_id] = 1

    for site in sites:
        subjects = os.listdir(os.path.join(config.DATA_DIR, 'Site' + str(site)))
        for subject in tqdm(subjects):
            session_dir = os.path.join(config.DATA_DIR, 'Site' + str(site), subject)
            subject_id = subject[:(len(subject)-1)/2]
            if os.path.isdir(session_dir):
                sessions = os.listdir(session_dir)
                for session in sessions:
                    data_folder = os.path.join(session_dir, session)
                    img_array = load_scan(data_folder, subject_id, session)

                    if img_array is not None: 
                      print ('Image Shape', img_array.shape, 'Image max: ', np.max(img_array), 'Image min: ', np.min(img_array), 'Image mean: ', np.mean(img_array))
                      x = infinite_generator_from_one_volume(config, img_array)
                      
                      labels = [subject_label_mapping[subject_id]]*config.scale

                      if x is not None:
                          slice_set.extend(x)
                          labels_set.extend(labels)
                        
    return np.array(slice_set), np.array(labels_set)



print(">> Fold {}".format(fold))
cube, cube_labels = get_self_learning_data([fold], config)
print("cube: {} | {:.2f} ~ {:.2f}".format(cube.shape, np.min(cube), np.max(cube)))

np.save(os.path.join(options.save, 
                     "batch_CT_"+str(config.scale)+
                     "_"+str(config.input_rows)+
                     "x"+str(config.input_cols)+
                     "x"+str(config.input_deps)+
                     "_"+str(fold)+".npy"), cube)

np.save(os.path.join(options.save, 
                     "batch_CT_labels_"+str(config.scale)+
                     "_"+str(config.input_rows)+
                     "x"+str(config.input_cols)+
                     "x"+str(config.input_deps)+
                     "_"+str(fold)+".npy"), cube_labels)