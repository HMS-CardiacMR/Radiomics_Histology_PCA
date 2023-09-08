import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

import numpy as np
import pandas as pd


def extract_all_radiomics_features(all_img, all_mask,  voxelspacing=None, params=None, manualnormalize=False):

    fvector = []
    dumi = np.expand_dims(all_img[:, :], 2)
    dumm = np.expand_dims(all_mask[:, :], 2)

    my_img = sitk.GetImageFromArray(dumi)
    my_msk = sitk.GetImageFromArray(dumm)

    dumrad = runpyradiomicsonimage(my_img, my_msk, voxelspacing, params=params, manualnormalize=manualnormalize)
    dumrad_val = list(dumrad.values())
    for i in range(36):# first 22 entries are 'diagnostics variables' by pyradiomics; i.e not image features so exclude
                        # 14 entries are shape features
        dumrad_val.pop(0)

    fvector = np.concatenate((np.asarray(dumrad_val, dtype=np.float64), fvector), axis=0)

    return fvector

def runpyradiomicsonimage(img, mask, voxelspacing, params=None, manualnormalize=False):
    # Normalize by 0.99 percentile
    if manualnormalize:
        dumim = sitk.GetArrayFromImage(img)
        dumsk = sitk.GetArrayFromImage(mask)
        maskedimg = np.multiply(dumim, dumsk)
        maskedvalues = dumim[np.where(dumsk == 1)]
        minimg = np.quantile(maskedvalues, 0.01)
        maximg = np.quantile(maskedvalues, 0.99)
        target_max_img = 255
        imagenorm = (maskedimg - minimg) * target_max_img / (maximg - minimg)
        imagenorm[np.where(imagenorm > target_max_img)] = target_max_img
        imagenorm[np.where(imagenorm < 0)] = 0
        imagesitk = sitk.GetImageFromArray(imagenorm)
    else:
        imagesitk = img
    #prepare the image and mask into SimpleITK
    VoxelSpacing = np.transpose(voxelspacing.astype('float'))
    imagesitk.SetSpacing(VoxelSpacing)
    imagesitk.SetOrigin(np.zeros(np.shape(VoxelSpacing)))

    masksitk = mask
    masksitk.SetSpacing(VoxelSpacing)
    masksitk.SetOrigin(np.zeros(np.shape(VoxelSpacing)))

    # Prepare the settings for the pyradiomics feature extractor
    settings = {}
    settings['normalize'] = False
    settings['interpolator'] = 'sitkBSpline'
    settings['verbose'] = True
    settings['force2D'] = True
    settings['removeOutliers'] = False
    # first order specific settings:
    settings[
        'voxelArrayShift'] = 0  # Minimum value in HU is -1000, shift +1000 to prevent negative values from being squared.
    settings['distances'] = [1]
    settings['weightingNorm'] = 'no_weighting'
    settings['symmetricalGLCM'] = True

    # Here overwrite all the settings parameters with the possible input kwargs
    accepted_settings_args = {'normalize', 'binWidth', 'resampledPixelSpacing', 'interpolator', 'verbose', 'force2D',
                              'voxelArrayShift', 'distances', 'weightingNorm', 'symmetricalGLCM', 'removeOutliers',
                              'binCount'}
    for key in params:
        if key in accepted_settings_args:
            settings[key] = params[key]

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()


    if 'featureclass' in params.keys():
        featureclass = params['featureclass']
        for feature in featureclass:
            extractor.enableFeatureClassByName(feature)
    else:
        # extractor.enableFeatureClassByName('firstorder')
        # extractor.enableFeatureClassByName('glcm')
        # extractor.enableFeatureClassByName('gldm')
        # extractor.enableFeatureClassByName('glrlm')
        # extractor.enableFeatureClassByName('glszm')
        # extractor.enableFeatureClassByName('ngtdm')
        # extractor.enableFeatureClassByName('shape2D')
        # extractor.enableFeatureClassByName('LoG', enabled=True, customArgs={'sigma': [1, 2, 3]})
        extractor.enableAllFeatures()

    if 'imagetypes' in params.keys():
        imagetypes = params['imagetypes']
        for imagetype in imagetypes:
            if imagetype == 'LoG':
                extractor.enableImageTypeByName('LoG', enabled=True, customArgs={'sigma': [1, 2, 3]})
            else:
                extractor.enableImageTypeByName(imagetype)
    else:
        extractor.enableAllImageTypes()


    radiomics.setVerbosity(50)  # Only critical warnings
    features = extractor.execute(imagesitk, masksitk, label=1)


    return features