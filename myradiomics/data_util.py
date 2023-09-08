import numpy as np
import scipy.io as sio
# import matplotlib.pyplot as plt
from PIL import Image as pillow_im
# from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import normalize


def extract_uncorrelated(dataset, threshold):
    corr_matrix = dataset.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    col_to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return dataset.drop(columns=col_to_drop), col_to_drop

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

def trimm_correlated(df_in, threshold):
    df_corr = df_in.corr(method='pearson', min_periods=1)
    df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)]*2, dtype=bool))).abs() >= threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = df_in[un_corr_idx]
    return df_out

def plotAUC(fpr, tpr, roc_auc, legend_txt='', init_plot=False, width=1, legend=True):
    if legend:
        plt.plot(fpr, tpr, lw=width, label= legend_txt+": AUC = %0.2f)" % roc_auc)
    else:
        plt.plot(fpr, tpr, lw=width)

    if init_plot==True:
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.gca().set_aspect('equal', 'box')
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.title('Testing Dataset')


from sklearn.model_selection import StratifiedShuffleSplit
def split_data(dtable, test_size, stratify_by, random_state):
    test_size = round(test_size*100)/100
    strtify_var = dtable[stratify_by]
    if test_size ==0:
        trn_tableIDX = np.arange(dtable.shape[0])
        tst_tableIDX = np.arange(dtable.shape[0])
    else:
        sss = StratifiedShuffleSplit(n_splits=2, train_size=1 - test_size, test_size=test_size, random_state=random_state)
        dum1, dum2 = sss.split(strtify_var, strtify_var)
        trn_tableIDX = dum1[0]
        tst_tableIDX = dum1[1]

    return dtable.iloc[trn_tableIDX].copy(), dtable.iloc[tst_tableIDX].copy()
