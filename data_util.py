import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image as pillow_im

'''
DUMMY DocString
'''
def plotAUC(fpr, tpr, roc_auc, legend_txt='', init_plot=False, width=1, legend=True):
    if legend:
        plt.plot(fpr, tpr, lw=width, label= legend_txt+": AUC = %0.2f)" % roc_auc)
    else:
        plt.plot(fpr, tpr, lw=width)

    if init_plot==True:
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.01])
        plt.ylim([0.0, 1.01])
        plt.gca().set_aspect('equal', 'box')
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.title('Testing Dataset')

def correl(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

def load_data_ACC(dtable,value4nan=None):
    risk_factors = ['emr_famSCD', 'emr_mri_mwtGT30', 'emr_sync', 'emr_nsvt',
                    'emr_lgeGT15', 'emr_mri_LVEFless50', 'emr_apicAneurysm' ]

    dum = dtable[risk_factors].copy()

    if value4nan.any()==None:
        value4nan = dum.median(axis=0)

    for i,col in enumerate(dum.columns):
        dum[col] = dum[col].fillna(value4nan[col])

    x = np.asarray(dum).astype(np.float64)

    return x

def load_dataESC(dtable,value4nan=None):
    dum = dtable[['emr_mwt_echo','emr_sizeLA_echo', 'maxLVOTg_rest_n_vals',
                  'emr_famSCD','emr_nsvt', 'emr_sync', 'emr_ageCMR'
                  ]].copy()

    if value4nan.any()==None:
        value4nan = dum.median(axis=0)

    for i,col in enumerate(dum.columns):
        dum[col] = dum[col].fillna(value4nan[col])

    x = np.asarray(dum).astype(np.float64)

    return x

from sklearn.preprocessing import normalize, StandardScaler
def NormalizeData(data, feats_axis=1, norm_type='unit_max', return_limits=False):
                                    # 'unit_max' or 'none'  'unit_vector'
    if norm_type == 'none':
        data = data
    elif norm_type == 'unit_vector':
        data = normalize(data)
    elif norm_type == 'std_norm':
        scaler = StandardScaler().fit(data)
        data   = scaler.transform(data)
    elif norm_type == 'unit_max':
        NR=data.shape[feats_axis]
        minv = np.zeros((NR,1))
        maxv = np.zeros_like(minv)
        for i in range(data.shape[feats_axis]):
            if feats_axis == 1:
                vector = np.nan_to_num(data[:,i], np.nanmedian(data[:,i], 0)) # just in case there is a NAN in the vector
                minv[i]=np.nanquantile(vector, 0.05)
                maxv[i]=np.nanquantile(vector, 0.95)
                data[:,i] = (vector - np.nanquantile(vector, 0.05)) / (
                            np.nanquantile(vector, 0.95) - np.nanquantile(vector, 0.05) + 0.0000000001)
            elif feats_axis ==0:
                vector = data[i,:]
                data[i,:] = (vector - np.nanquantile(vector,0.05)) / (np.nanquantile(vector,0.95) - np.nanquantile(vector,0.05) + 0.0000000001)

            else:
                print('ERROR: Data Normalization Function Not Defined for Features On Axis: ', feats_axis)

        data[data<0] = 0
        data[data>1] = 1

    if return_limits:
        return data, minv, maxv
    else:
        return data


def filter_data_scd(dtable, min_fu_dur=90, include_site='all',cohort='all',exclude_site = 'none', contour_type = 'manual_only'):

    if (include_site != 'all') & (exclude_site != 'none'):
        print('ERROR: Inclusion & exclusion of sitesusing the same filter is NOT allowed. Please, use 2 separate filters.')
        return

    incl = np.zeros((1,dtable.shape[0]),dtype=bool)
    incl = incl | (np.asarray(dtable['outcome_fu_duration']>min_fu_dur))
    print(np.count_nonzero(incl))

    dum = ~np.asarray(dtable['mat_fn'].str.contains("no_match")) # remove cases without matched LGE images
    dum[dtable['priorOOHCA']==1] = 0 # patient excluded for proir OOHCA

    if contour_type == 'manual_only':# remove cases not refined after CNN segmentation
        dum = dum & ~np.asarray(dtable['mat_fn'].str.contains("CNN"))
    elif contour_type == 'cnn_only': # independent dataset
        dum = dum & np.asarray(dtable['mat_fn'].str.contains("CNN"))
        # dum = dum & ~np.asarray(dtable['mat_fn'].str.contains("oohca")) # patient excluded for proir OOHCA
    # elif contour_type == 'all':
    #     dum = dum & ~np.asarray(dtable['mat_fn'].str.contains("oohca")) # patient excluded for proir OOHCA

    incl = incl & dum

    if include_site != 'all':
        dum = np.asarray(dtable['mat_fn'].str.contains(include_site))  # remove cases without matched LGE images
        incl = incl & dum

    if exclude_site != 'none':
        dum = ~np.asarray(dtable['mat_fn'].str.contains(include_site))  # remove cases without matched LGE images
        incl = incl & dum

    if cohort != 'all':
        dum = np.asarray(dtable['emr_lvot_obstruction']== int(cohort=='obstructive'))  # remove cases without matched LGE images
        incl = incl & dum

    print(np.count_nonzero(incl))

    filtered_table  = dtable[np.transpose(incl)].copy()
    excluded_records= dtable[np.transpose(~incl)].copy()

    return filtered_table, excluded_records

## The following code is not needed.... just for illustration purpose

def load_images(mat_fn,data_type='whole_lge_image',out_imsize=[128,128,-1],
                                                 spatial_norm_flag=True, intensity_norm_scope='none'):
    # LOAD MAT FILES.....
    image_volumes = []
    num_sl_pat = 0
    for i, im_fn in enumerate(mat_fn):

        dd = sio.loadmat(im_fn, mat_dtype=True)
        pxsize = dd['pixdim'][0]
        if pxsize[0]>10: # data export error
            pxsize[0:2]=1

        epi_maskVol = dd['epi_masks']
        myo_maskVol = dd['myo_masks']
        lge_imgVol  = dd['lge_images']

        if epi_maskVol.shape[-1] < lge_imgVol.shape[-1]:
            print('epi size = ', epi_maskVol.shape[-1])
            print('lge size = ', lge_imgVol.shape[-1])
            print(im_fn)
            continue

        if data_type == 'whole_lge_image':
            tmp_imvol = lge_imgVol
        elif data_type == 'epi_lge_roi':
            tmp_imvol = np.multiply(epi_maskVol, lge_imgVol)
        elif data_type == 'myo_lge_roi':
            tmp_imvol = np.multiply(myo_maskVol, lge_imgVol)
        elif data_type == 'myo_mask':
            tmp_imvol = myo_maskVol

        if spatial_norm_flag:
            tmp_imvol = normalize_pxsize(tmp_imvol, insz=pxsize[0:2], outsz=[1,1])

        if tmp_imvol.shape[0] > out_imsize[0]:
            tmp_imvol = crop_vol(tmp_imvol,axs=0,outsz=out_imsize[0],crop_center='com')
        elif tmp_imvol.shape[0] < out_imsize[0]:
            tmp_imvol = pad_vol(tmp_imvol,axs=0,outsz=out_imsize[0],crop_center='com')

        if tmp_imvol.shape[1] > out_imsize[1]:
            tmp_imvol = crop_vol(tmp_imvol,axs=1,outsz=out_imsize[1],crop_center='com')
        elif tmp_imvol.shape[1] < out_imsize[1]:
            tmp_imvol = pad_vol(tmp_imvol,axs=1,outsz=out_imsize[1],crop_center='com')

        if intensity_norm_scope == 'volume': # normalize Global
            norm_min = np.min(tmp_imvol)
            norm_max = np.max(tmp_imvol)
            tmp_imvol = (tmp_imvol - norm_min) / ((norm_max - norm_min) + 0.0001)
        elif intensity_norm_scope == 'slice':
            for idx in range(tmp_imvol.shape[-1]):  # loop on slices/channels....
                # Extract one slice-image; per-channel normalization
                tmp_im = np.squeeze(tmp_imvol[:, :, idx])
                tmp_im = tmp_im.astype(np.float64)
                norm_min = np.min(tmp_im)
                norm_max = np.max(tmp_im)
                tmp_imvol[:, :, idx] = (tmp_im - norm_min) / ((norm_max - norm_min) + 0.0001)

        image_volumes.append(tmp_imvol)

    return np.asarray(image_volumes)

def crop_vol(imvol,axs,outsz,crop_center='com'):
    tmp_imvol = []
    for i in range(imvol.shape[-1]):
        img = imvol[:,:,i]
        if crop_center == 'com': #center of mass
            proj = np.nonzero(np.sum(img, axis=1-axs) > 1)[0]
            im_cntr = np.round(np.average(proj))
        else:
            im_cntr = img.shape[axs] // 2
        si =  np.int16(im_cntr- outsz // 2)

        if axs==0:
            tmp_imvol.append(img[si:si + outsz,:])
        elif axs==1:
            tmp_imvol.append(img[   :  ,  si:si + outsz])

    return np.moveaxis(np.asarray(tmp_imvol),0,-1) # put slices in last dimension

def pad_vol(imvol,axs,outsz,crop_center='com'):

    if len(imvol.shape)==3:
        if axs==0:
            tmp_imvol = np.zeros((outsz,imvol.shape[1],imvol.shape[2]))
        elif axs==1:
            tmp_imvol = np.zeros((imvol.shape[0],outsz,imvol.shape[2]))
    else:
        if axs==0:
            tmp_imvol = np.zeros((outsz,imvol.shape[1]))
        elif axs==1:
            tmp_imvol = np.zeros((imvol.shape[0],outsz))

    if crop_center == 'com':  # center of mass
        proj = np.nonzero(np.sum(imvol, axis=axs) > 10)[0]
        vol_cntr = np.round(np.average(proj))
    else:
        vol_cntr = imvol.shape[axs] // 2

    si = np.int16(vol_cntr - imvol.shape[axs] // 2)
    ds = np.int16(imvol.shape[axs] // 2)

    if axs==0:
        tmp_imvol[si:si + ds, :,   :] = imvol
    elif axs==1:
        tmp_imvol[:, si:si + ds,   :] = imvol

    return tmp_imvol


def normalize_pxsize(imvol, insz, outsz=[1,1]):
    norm_imvol = []
    for i in range(imvol.shape[-1]):
        #dum = cv2.resize(imvol[:,:,i],None, fx=insz[0],fy=insz[1], interpolation=cv2.INTER_LINEAR) # bilinear interpolation
        wnew = int(insz[0] / outsz[0]*imvol.shape[0])
        hnew = int(insz[1] / outsz[1] * imvol.shape[1])
        dum_im = pillow_im.fromarray(imvol[:, :, i])
        dum_im = dum_im.resize((wnew,hnew), resample=pillow_im.BILINEAR)
        norm_imvol.append(np.array(dum_im))

    return np.moveaxis(np.asarray(norm_imvol),0,-1) # put slices in last dimension

def fillin_stack(imvol,depth): #not tested yet
    cur_depth = imvol.shape[-1]
    cur_center = np.int16(np.round(cur_depth/2))
    d = depth - cur_depth
    dum = imvol[:,:,0:cur_center+1]
    for i in range(d):
        dum= np.append(dum,np.expand_dims(imvol[:,:,cur_center],axis=-1),axis=2)
    return np.concatenate((dum,imvol[:,:,cur_center:-1]), axis=2)