# Unveiling the Hidden Insights: Radiomics Signature of the Non-Ischemic Dilated Cardiomyopathy Using Cardiovascular Magnetic Resonance

## Description of Code

This python code implements our PCA logistic regression models to identify components of the extracellular matrix in dilated cardiomyopathy.
 
First, radiomic features should be extracted using compute_radiomic_features.py in myradiomics. This requires the installation of pyradiomics library.
The data should be in .mat format with:
-First matrix reprensts the image
-Second matrix represents the mask
-The pixel spacing should be stored in px_size variable.

The code will read the images and masks and return the computed radiomic features. Note that the shape features are excluded due to the nature of the study. Since the region of interest near the biopsy regino does not provide any useful information about the shapes.
The code will loop over all sequences and return all radiomic features per sequence.


Run PCA_analysis for principla component analysis. The code will also return radiomic features that are highly correlated with each component.


## Abstract

Background: Current cardiovascular magnetic resonance (CMR) sequences cannot discriminate between different myocardial extracellular space (ECS), including collagen, non-collagen, and inflammation. We sought to investigate if CMR radiomics analysis can distinguish between non-collagen and inflammation from collagen in dilated cardiomyopathy (DCM).

Methods: In a retrospective study, we identified data from 132 DCM patients scheduled for an invasive septal biopsy who underwent CMR at 3T. CMR imaging protocol included native and post-contrast T1 mapping and late gadolinium enhancement (LGE). Radiomic features were computed from the mid-septal myocardium, near the biopsy region, on native T1, extracellular volume (ECV) map, and LGE images. Principal component analysis was used to reduce the number of radiomic features to five principal radiomics. Moreover, a correlation analysis was conducted to identify radiomic features exhibiting a strong correlation (r >0.9) with the five principal radiomics. Biopsy samples were used to quantify ECS, myocardial fibrosis, and inflammation.

Results: Four histopathological phenotypes were identified as normal (n=20), non-collagenous ECS expansion (n=49), collagenous ECS expansion (n=42), and severe fibrosis (n=21). Non-collagenous expansion was associated with the highest risk of myocardial inflammation (65%). While native T1 and ECV provided high diagnostic performance in differentiating severe fibrosis (C-statistic: 0.90 and 0.90, respectively), their performance in differentiating between non-collagen and collagenous expansion decreased (C-statistic: 0.59 and 0.55, respectively). Integration of ECV principal radiomics provided better discrimination and reclassification between non-collagen and collagen (C-statistic: 0.79, net reclassification index (NRI) 0.83; 95% CI 0.45-1.22, p<0.001). There was a similar trend in the addition of native T1 principal radiomics (C-statistic: 0.75, NRI 0.93; 95%CI 0.56-1.29, p<0.001) and LGE principal radiomics (C-statistic: 0.74, NRI 0.59; 95%CI 0.19-0.98, p=0.004). Five radiomic features per sequence were ide
ntified using correlation analysis. They showed a similar improvement in performance for differentiating between non-collagen and collagen (native T1, ECV, LGE C-statistic: 0.75, 0.77, and 0.71, respectively). These improvements remained significant when confined to a single radiomic feature (native T1, ECV, LGE C-statistic: 0.71, 0.70, and 0.64, respectively).

Conclusions: Radiomic features extracted from native T1, ECV, and LGE provide incremental information that improves our capability to discriminate non-collagenous expansion from collagen and could be useful for detecting subtle chronic inflammation in DCM patients.
