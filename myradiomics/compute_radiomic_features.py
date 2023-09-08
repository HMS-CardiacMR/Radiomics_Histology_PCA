"""
This script extracts radiomics features
"""
import os
import numpy as np
import scipy.io
import random
from radiomics_utilities import extract_all_radiomics_features
import pandas as pd



random.seed(2022)


data_dir = 'add path to the data here'

feats = pd.DataFrame()
patient_rad_feats = pd.DataFrame()

radiomics_params = {}
radiomics_params['binWidth'] = 1

csvfn_basename = 'path to where you save the radiomic features'


columns_labels = pd.read_csv("Name of the features extracted from pyradiomics library")

columns_labels = list(columns_labels.columns.values)

# patient dir list
pat_list = os.listdir(data_dir)

matfn_dict = {'t1': '/t1_image_and_mask.mat',  'ecv': '/ecv_map_and_mask.mat', 'lge': '/lge_image_and_mask.mat'           
              }
available_sequences = ['t1', 'ecv', 'lge']

for seq_type in available_sequences: # loop on all sequences considered in this study
    csvfn = csvfn_basename + '/radiomics_4biopsy_all_features_' + seq_type + '.csv'


    feats = pd.DataFrame()
    for idx, pat_dn in enumerate(pat_list): # loop on all patients and find the desired sequence

        flist = os.listdir(data_dir + pat_dn)
        if len([1 for fn in flist if
                fn.startswith(seq_type + '_')]) == 0:  # no file containing images and masks for this sequence
            continue  # #skip it
        else:
            mat_fn = data_dir + pat_dn + matfn_dict[seq_type]
            
            #The data is saved in .mat format where the first matrix is the image and the second one is the mask
            data = sio.loadmat(mat_fn)
            image_n_mask = data.get("image_n_mask")
            pxsize = data['px_size'][0]
            
            image = image_n_mask[:, :, 0]
            image = np.squeeze(normalize_pxsize(image, insz=pxsize[0:2], outsz=[1, 1]))

            norm_min = np.min(image)
            norm_max = np.max(image)
            image = (image - norm_min) / ((norm_max - norm_min) + 0.0001)
            
            mask = image_n_mask[:, :, 1]
            mask = np.squeeze(normalize_pxsize(mask, insz=pxsize[0:2], outsz=[1, 1]))
            
            patient_rad_feats = extract_all_radiomics_features(all_img=the_images, all_mask=the_masks,
                                                               voxelspacing=np.asarray([1., 1., 1.]),
                                                               params=radiomics_params,
                                                               manualnormalize=True)  


            SR1 = pd.Series(patient_rad_feats, name=idx)
            SR2 = pd.Series([pat_dn], name=idx)
            SR = pd.concat([SR2, SR1], ignore_index=True)
            SRDF = pd.DataFrame(SR).transpose()
            feats = feats.append(SRDF, verify_integrity=True)
            feats.to_csv(csvfn, index=False)
            print('Done saving patient ', idx, '____', int(idx / 100) / 10)

    feats.set_axis(columns_labels, axis=1, inplace=True)
    feats.to_csv(csvfn, index=False)
    print(feats.head(3))
    del feats
