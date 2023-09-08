from myradiomics.data_util import NormalizeData, trimm_correlated
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import clone
from numpy.random import seed
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.patheffects as PathEffects

pca_seed = 1964
seed(pca_seed)
path = "path to the data"

data = pd.read_csv(path)
sequence = "T1, ECV, lge"
classes = "23"
idx = np.nonzero(np.all(np.asarray(data) == 0, axis=0))[
    0]
data.drop(columns=data.columns[idx], axis=1, inplace=True)

desired_categories = [2, 3]
exclude_histopath_categories = [1, 4]
idx = []
for ct in exclude_histopath_categories:
    idx.append(data[data["Histopathology_label"] == ct].index)
idx = np.concatenate(idx, axis=0)
data.drop(np.asarray(idx), axis=0, inplace=True)
n_classes = 2
original_labels = np.asarray(data.loc[:, "Histopathology_label"])
labels = np.asarray(data.loc[:, "Histopathology_label"])
patients = np.asarray(data.loc[:, "pat_id"])
data.drop('pat_id', axis=1, inplace=True)
data.drop('Histopathology_label', axis=1, inplace=True)

data.drop(columns=data.columns[idx], axis=1, inplace=True)
data = trimm_correlated(data, 0.80)
print(data.head())

trainX = np.asarray(data.iloc[:, :])
trainX = NormalizeData(trainX, feats_axis=1,
                       norm_type='unit_max')


tsne = TSNE(random_state=1964, n_components=2)
tsne_results = tsne.fit_transform(trainX)
print(tsne_results.shape)

def plot_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], c=palette[colors.astype(np.int)], cmap=plt.cm.get_cmap('Paired'))
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    # ax.axis('off')
    ax.axis('tight')  # add the labels for each digit corresponding to the label
    txts = []
    for i in range(num_classes):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    return f, ax, sc, txts

histology_labels = np.where(labels==2, 0, 1)
# f, ax, sc, txts = plot_scatter(tsne_results, histology_labels)
# # ax.grid()
# f.show()

maxN_rads = 5

# transformer = PCA(n_components=maxN_rads, random_state=pca_seed, whiten=True)
transformer = PCA(0.9, random_state=pca_seed, whiten=True)
rad_features_trn = transformer.fit_transform(trainX)
rad_features_tst = transformer.transform(trainX)

print("Explained variance: ", transformer.explained_variance_)
print("Explained variance ratio: ", transformer.explained_variance_ratio_)
# print("Explained variance : ", transformer.explained_variance_ratio_.cumsum())
variance = transformer.explained_variance_ratio_
for element in variance:
    print(element)
exit()
AUCmatrix = np.zeros((1, maxN_rads))
cutoffMatrix = np.zeros((1, maxN_rads))
fusClf_0 = LogisticRegression(random_state=pca_seed, n_jobs=-1, max_iter=500, verbose=False, class_weight='balanced')
labels = label_binarize(labels, classes=desired_categories)  #
for model_n_rad in range(1, maxN_rads+1):  # this is the number of rads; will subtract 1 for index below
    ## maximum correlated Radiomic feature witht the PCA radiomics
    xx = rad_features_trn[:, model_n_rad - 1].reshape(1, -1)
    d = [cosine_similarity(xx, trainX[:, rad_i].reshape(1, -1)) for rad_i in
         range(trainX.shape[-1])]
    dm = np.max(d)
    ddm = np.where(d >= 0.9 * dm)[0]
    d = np.asarray(d)
    d = d.reshape(1, 114)
    print(d.shape)

    plt.imshow(d, cmap="plasma", aspect="auto")
    plt.axis("off")
    plt.colorbar()
    plt.show()
    exit()
    ColNames = list(data.columns)
    [print('PCA Radiomic #', model_n_rad, ' corresponds to Radiomics: ', ColNames[ii]) for ii in ddm]

    print('####################  SINGLE RADIOMICS MODELS  ####################', 'N = ', str(model_n_rad))
    print('#####################################################################')
    fusClf = clone(fusClf_0)  # class_weight is implecitly applied: 'balanced'
    fusClf = OneVsRestClassifier(fusClf)
    fusClf.fit(np.expand_dims(rad_features_trn[:, model_n_rad - 1], axis=1),
               labels)  # measure power of a single RAD
    prd_rad_tst = fusClf.decision_function(np.expand_dims(rad_features_tst[:, model_n_rad - 1], axis=1))
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if n_classes > 2:
        for c in range(n_classes):
            fpr[c], tpr[c], _ = roc_curve(labels[c], prd_rad_tst[c])
            roc_auc[c] = auc(fpr[c], tpr[c])
            print('Testing--RADIOMICS, Single RAD {i}, Class {j}: AUC = {AUC}'.
                  format(i=model_n_rad, j=c, AUC=roc_auc[c]))
    else:
        fpr[1], tpr[1], _ = roc_curve(labels, prd_rad_tst)
        roc_auc[1] = auc(fpr[1], tpr[1])
        print('Testing--RADIOMICS, Single RAD {i}, Class {j}: AUC = {AUC}'.format(i=model_n_rad, j=1, AUC=roc_auc[1]))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), prd_rad_tst.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print('Testing--RADIOMICS, Single RAD {i}, MICRO: AUC = {AUC}'.format(i=model_n_rad, AUC=roc_auc["micro"]))

    print('####################  MULTIPLE RADIOMICS MODELS  ####################', 'N = ', str(model_n_rad))
    print('#####################################################################')
    fusClf = clone(fusClf_0)  # class_weight is implecitly applied: 'balanced'
    fusClf = OneVsRestClassifier(fusClf)
    fusClf.fit(rad_features_trn[:, :model_n_rad], labels)  # measure power of a single RAD
    prd_rad_tst = fusClf.decision_function(rad_features_tst[:, :model_n_rad])
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if n_classes > 2:
        for c in range(n_classes):
            fpr[c], tpr[c], _ = roc_curve(labels[:, c], prd_rad_tst[:, c])
            roc_auc[c] = auc(fpr[c], tpr[c])
            print('Testing--RADIOMICS, MULTIPLE RAD {i}, Class {j}: AUC = {AUC}'.format(i=model_n_rad,
                                                                                        j=desired_categories[c],
                                                                                        AUC=roc_auc[c]))
    else:
        fpr[1], tpr[1], _ = roc_curve(labels, prd_rad_tst)
        roc_auc[1] = auc(fpr[1], tpr[1])
        print('Testing--RADIOMICS, MULTIPLE RAD {i}, Class {j}: AUC = {AUC}'.format(i=model_n_rad,
                                                                                    j=desired_categories[1],
                                                                                    AUC=roc_auc[1]))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), prd_rad_tst.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print('Testing--RADIOMICS, MULTIPLE RAD {i}, MICRO: AUC = {AUC}'.format(i=model_n_rad, AUC=roc_auc["micro"]))

## MACRO
if n_classes > 2:
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw = 2
plt.figure(figsize=(10, 10))
plt.gca().set_aspect('equal', 'box')

if n_classes > 2:
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
colors = cycle(["aqua", "darkorange", "cornflowerblue", "magenta"])
if n_classes > 2:
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="Class {0} (area = {1:0.2f})".format(desired_categories[i], roc_auc[i]),
        )
else:
    plt.plot(fpr[1], tpr[1], color="darkorange", lw=lw,
             label="Class {0} (area = {1:0.2f})".format(desired_categories[1], roc_auc[1]),
             )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
# plt.title("Receiver operating characteristic to multiclass", fontsize=14)
plt.legend(loc="lower right", fontsize=14)
plt.savefig("path to save results")