from config_sukrit_MRI import models_genesis_config
import numpy as np
import os
from tqdm import tqdm
import nibabel as nib
from utils import *

def get_self_learning_data(fold, config):
	slice_set = []
	slice_set_segmentation_masks = []
	sites = fold
	for site in sites:
		subjects = os.listdir(os.path.join('../../ATLAS_data', 'Site' + str(site)))
		for subject in tqdm(subjects):
			session_dir = os.path.join('../../ATLAS_data', 'Site' + str(site), subject)
			if os.path.isdir(session_dir):
				sessions = os.listdir(session_dir)
				for session in sessions:
					data_folder = os.path.join(session_dir, session)

					subject_file_name = subject + '_t1w_deface_stx.nii.gz'
					img = nib.load(os.path.join(data_folder, subject_file_name))
					img_array = img.get_fdata()

					segmentation_true_file_name = subject + '_LesionSmooth_stx.nii.gz'
					segmentation_true = nib.load(os.path.join(data_folder, segmentation_true_file_name))
					segmentation_true_array = segmentation_true.get_fdata()

					hu_max = 100
					hu_min = 0
					
					HU_thred = (33 - hu_min) / (hu_max - hu_min)

					img_array[img_array < hu_min] = hu_min
					img_array[img_array > hu_max] = hu_max
					img_array = 1.0*(img_array-hu_min) / (hu_max-hu_min)
					img_array = img_array

					img_array = np.expand_dims(np.array(img_array), axis=0)
					img_array = np.expand_dims(np.array(img_array), axis=0)

					#img_array = img_array.transpose(2, 1, 0)
					print ('\n Image Shape', img_array.shape, 'Image max: ', np.max(img_array), 'Image min: ', np.min(img_array), 'Image mean: ', np.mean(img_array))

					print ('Segmentation True Shape', segmentation_true_array.shape, 'Segmentation Image max: ', np.max(segmentation_true_array), 'Segmentation Image min: ', np.min(segmentation_true_array), ' Segmentation Image mean: ', np.mean(segmentation_true_array))

					training_generator = generate_pair(img_array, 1, config, "train")
					image, gt = next(training_generator)
					image, gt = next(training_generator)
					image, gt = next(training_generator)
					image, gt = next(training_generator)
					image, gt = next(training_generator)
					image, gt = next(training_generator)

					'''
					x, mask = infinite_generator_from_one_volume(config, img_array, segmentation_true_array)

					if x is not None and mask is not None:
						slice_set.extend(x)
						slice_set_segmentation_masks.extend(mask)
					'''

config = models_genesis_config()

print(">> Fold {}".format(0))
cube, segmentation_cube = get_self_learning_data([0], config)


