import os
import shutil
import numpy as np
os.chdir('./')

print ('CWD:', os.getcwd())

base_dir = '../modified_data/'
subjects = os.listdir(base_dir)


### Redo the directory structure for CQ500 data #############

### remove spaces from the folder names and save in a new location ###
for old_subject in subjects: 
	new_subject = old_subject.replace(" ", "_")
	session_dir = os.path.join(base_dir, old_subject, 'Unknown Study')
	if os.path.isdir(session_dir):
		sessions = os.listdir(session_dir)
		for old_session in sessions:
			new_session = old_session.replace(" ", "_")
			new_data_folder = os.path.join('../modified_data/', new_subject, new_session)
			print ('Folder: ', new_data_folder)
			if not os.path.isdir(new_data_folder):
				shutil.copytree(os.path.join(session_dir, old_session), new_data_folder)


### Convert from DICOM to NIFTI using dcm2niix command ###############
'''
for subject in subjects:
	session_dir = os.path.join(base_dir, subject,)
	if os.path.isdir(session_dir):
		sessions = os.listdir(session_dir)
		for session in sessions:
			data_folder = os.path.join(session_dir, session)
			print ('Folder: ', data_folder)
			os.system('dcm2niix -z y -f %i_\%f_\%p_\%t_\%s -m y -o ' + './preprocessed_CT/' + ' ' + data_folder)
'''