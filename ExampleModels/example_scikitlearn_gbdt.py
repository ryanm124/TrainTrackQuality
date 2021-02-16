# Scikit-Learn and skl2onnx must be installed to use this script
# Go to https://scikit-learn.org/stable/install.html and 
# https://pypi.org/project/skl2onnx/ to install

import uproot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from functions import get_eff_faker_err, train_test_split_by_part, get_eff_faker_vs_feat
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

# -----IMPORT .ROOT FILES-----

# Take in .root into arrays
arrays = (uproot.open("TTbar_PU200_D49_prompt.root")["L1TrackNtuple/eventTree"]
                .arrays("*", namedecode="utf-8"))

# A look at what variables you can use as inputs to your model
arrays.keys()

# -----CREATE TRAIN AND TEST SETS-----

# Select features from data and put in proper format
features = ['trk_pt','trk_eta','trk_phi','trk_z0','trk_chi2rphi','trk_chi2rz','trk_bendchi2','trk_bendchi2']
X = np.empty((len(arrays[features[0]].flatten()),len(features)))
for i in range(len(features)):
    X[:,i] = arrays[features[i]].flatten()

# 'trk_fake' is the track quality variable, truth data
y = arrays['trk_fake'].flatten()
y[y==2] = 1

# Grab pdgid for certain studies
pdgid = arrays['trk_matchtp_pdgid'].flatten()

# Get rid of any nan instances for training purposes (doesn't converge)
find_nan = np.argwhere(np.isnan(X))
X = np.delete(X, find_nan[:,0], 0)
y = np.delete(y, find_nan[:,0])
pdgid = np.delete(pdgid, find_nan[:,0])

# Create train and test sets with 2500 of each mu, elec, had, fake in train, rest in test
X_train, y_train, pdgid_train, X_test, y_test, pdgid_test = train_test_split_by_part(X,y,pdgid,2500,2500,2500,2500)

# -----TRAIN MODEL-----

# This creates GBDT with 500 trees that each have a max depth of 3
# Visit https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
#   for more information on how to configure GBDT
clf_GBDT = GradientBoostingClassifier(n_estimators=500, max_depth=3)
clf_GBDT.fit(X_train,y_train)

# -----EVALUATE MODEL-----

# Create roc curve with AUC (area under curve) value
y_pred = clf_GBDT.predict_proba(X_test)[:,1]
fpr, tpr, dt = roc_curve(y_test,y_pred)
auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr)
plt.xlabel('FPR',fontsize=14)
plt.ylabel('TPR',fontsize=14)
plt.title('AUC = '+str(auc),fontsize=14)
plt.savefig('ROC_curve.png')

# Plot TPR and FPR vs decision threshold (dt)
plt.plot(dt[1:],fpr[1:],label='FPR')
plt.plot(dt[1:],tpr[1:],label='TPR')
plt.xlabel('decision thresh.',fontsize=14)
plt.ylabel('metric',fontsize=14)
plt.legend(loc='best')
plt.savefig('TPR_FPR_vs_dt.png')

# Create TPR/FPR vs. pt
pt, eff, faker, err_eff, err_faker = get_eff_faker_vs_feat('pt',features,X_test,y_test,clf_GBDT)

plt.errorbar(pt,eff,yerr=err_eff,linestyle='None',fmt='.')
plt.xlabel('p$_{T}$ (GeV/c)',fontsize=14)
plt.ylabel('TPR',fontsize=14)
plt.savefig('TPR_vs_pt.png')

plt.errorbar(pt,faker,yerr=err_faker,linestyle='None',fmt='.')
plt.xlabel('p$_{T}$ (GeV/c)',fontsize=14)
plt.ylabel('FPR',fontsize=14)
plt.savefig('FPR_vs_pt.png')

# Create TPR/FPR vs. eta
pt, eff, faker, err_eff, err_faker = get_eff_faker_vs_feat('eta',features,X_test,y_test,clf_GBDT)

plt.errorbar(pt,eff,yerr=err_eff,linestyle='None',fmt='.')
plt.xlabel('$\eta$',fontsize=14)
plt.ylabel('TPR',fontsize=14)
plt.savefig('TPR_vs_eta.png')

plt.errorbar(pt,faker,yerr=err_faker,linestyle='None',fmt='.')
plt.xlabel('$\eta$',fontsize=14)
plt.ylabel('FPR',fontsize=14)
plt.savefig('FPR_vs_eta.png')

# -----SAVE MODEL TO .ONNX-----

# This will output a model.onnx file
initial_type = [('feature_input', FloatTensorType([1, len(features)]))]
onx = onnxmltools.convert.convert_sklearn(clf_GBDT, initial_types=initial_type)
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

