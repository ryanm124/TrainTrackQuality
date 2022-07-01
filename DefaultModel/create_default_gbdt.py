import uproot3
import numpy as np
import matplotlib.pyplot as plt
from classes import dataType
from functions import get_eff_faker_err, train_test_split_by_part, get_eff_faker_vs_feat
import xgboost as xgb
from sklearn.utils.random import sample_without_replacement
from sklearn.utils import resample, shuffle
from sklearn.metrics import roc_curve, roc_auc_score
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

# -----IMPORT DATA-----
# take in .root into arrays
arrays_hybrid = (uproot3.open("combinedsample_pu200.root")["L1TrackNtuple/eventTree"]
                 .arrays("*", namedecode="utf-8"))

# create data in proper format
data = dataType(arrays_hybrid)

# list the features we are using
data.trks.X_feats

# -----CREATE TRAIN AND TEST DATA SET-----

# collect all data that passes pt cuts
pt_cutl = 20
pt_cuth = 100
X = data.trks.X[np.logical_and(data.trks.pt>pt_cutl,
                               data.trks.pt<pt_cuth)]
y = data.trks.y[np.logical_and(data.trks.pt>pt_cutl,
                               data.trks.pt<pt_cuth)]
pdgid = abs(data.trks.pdgid[np.logical_and(data.trks.pt>pt_cutl,
                                           data.trks.pt<pt_cuth)])
pt = data.trks.pt[np.logical_and(data.trks.pt>pt_cutl,
                                 data.trks.pt<pt_cuth)]
eta = data.trks.eta[np.logical_and(data.trks.pt>pt_cutl,
                                   data.trks.pt<pt_cuth)]
chi2pdof = data.trks.chi2pdof[np.logical_and(data.trks.pt>pt_cutl,
                                             data.trks.pt<pt_cuth)]

# select 4000 of each particle type randomly
mu_idx = sample_without_replacement(len(X[pdgid==13]),4000,random_state=23)
elec_idx = sample_without_replacement(len(X[pdgid==11]),4000,random_state=23)
qcd_idx = sample_without_replacement(len(X[np.logical_and(pdgid>37,pdgid!=999)]),4000,random_state=23)
fake_idx = sample_without_replacement(len(X[pdgid==999]),4000,random_state=23)

# take those 4000 of each particle type and make it the training set
X_train = np.concatenate((X[pdgid==13][mu_idx],
                          X[pdgid==11][elec_idx],
                          X[np.logical_and(pdgid>37,pdgid!=999)][qcd_idx],
                          X[pdgid==999][fake_idx]))
y_train = np.concatenate((y[pdgid==13][mu_idx],
                          y[pdgid==11][elec_idx],
                          y[np.logical_and(pdgid>37,pdgid!=999)][qcd_idx],
                          y[pdgid==999][fake_idx]))
pdgid_train = np.concatenate((pdgid[pdgid==13][mu_idx],
                              pdgid[pdgid==11][elec_idx],
                              pdgid[np.logical_and(pdgid>37,pdgid!=999)][qcd_idx],
                              pdgid[pdgid==999][fake_idx]))
pt_train = np.concatenate((pt[pdgid==13][mu_idx],
                           pt[pdgid==11][elec_idx],
                           pt[np.logical_and(pdgid>37,pdgid!=999)][qcd_idx],
                           pt[pdgid==999][fake_idx]))
eta_train = np.concatenate((eta[pdgid==13][mu_idx],
                            eta[pdgid==11][elec_idx],
                            eta[np.logical_and(pdgid>37,pdgid!=999)][qcd_idx],
                            eta[pdgid==999][fake_idx]))
chi2pdof_train = np.concatenate((chi2pdof[pdgid==13][mu_idx],
                                 chi2pdof[pdgid==11][elec_idx],
                                 chi2pdof[np.logical_and(pdgid>37,pdgid!=999)][qcd_idx],
                                 chi2pdof[pdgid==999][fake_idx]))
X_train,y_train,pdgid_train,pt_train,eta_train,chi2pdof_train = shuffle(X_train,y_train,pdgid_train,pt_train,
                                                                        eta_train,chi2pdof_train,random_state=23)

# all other tracks make up testing set
X_test = np.concatenate((np.delete(X[pdgid==13],mu_idx,axis=0),
                         np.delete(X[pdgid==11],elec_idx,axis=0),
                         np.delete(X[np.logical_and(pdgid>37,pdgid!=999)],qcd_idx,axis=0),
                         np.delete(X[pdgid==999],fake_idx,axis=0)))
y_test = np.concatenate((np.delete(y[pdgid==13],mu_idx,axis=0),
                         np.delete(y[pdgid==11],elec_idx,axis=0),
                         np.delete(y[np.logical_and(pdgid>37,pdgid!=999)],qcd_idx,axis=0),
                         np.delete(y[pdgid==999],fake_idx,axis=0)))
pdgid_test = np.concatenate((np.delete(pdgid[pdgid==13],mu_idx,axis=0),
                             np.delete(pdgid[pdgid==11],elec_idx,axis=0),
                             np.delete(pdgid[np.logical_and(pdgid>37,pdgid!=999)],qcd_idx,axis=0),
                             np.delete(pdgid[pdgid==999],fake_idx,axis=0)))
pt_test = np.concatenate((np.delete(pt[pdgid==13],mu_idx,axis=0),
                          np.delete(pt[pdgid==11],elec_idx,axis=0),
                          np.delete(pt[np.logical_and(pdgid>37,pdgid!=999)],qcd_idx,axis=0),
                          np.delete(pt[pdgid==999],fake_idx,axis=0)))
eta_test = np.concatenate((np.delete(eta[pdgid==13],mu_idx,axis=0),
                           np.delete(eta[pdgid==11],elec_idx,axis=0),
                           np.delete(eta[np.logical_and(pdgid>37,pdgid!=999)],qcd_idx,axis=0),
                           np.delete(eta[pdgid==999],fake_idx,axis=0)))
chi2pdof_test = np.concatenate((np.delete(chi2pdof[pdgid==13],mu_idx,axis=0),
                                np.delete(chi2pdof[pdgid==11],elec_idx,axis=0),
                                np.delete(chi2pdof[np.logical_and(pdgid>37,pdgid!=999)],qcd_idx,axis=0),
                                np.delete(chi2pdof[pdgid==999],fake_idx,axis=0)))
X_test,y_test,pdgid_test,pt_test,eta_test,chi2pdof_test = shuffle(X_test,y_test,pdgid_test,pt_test,
                                                                  eta_test,chi2pdof_test,random_state=23)

# bin all of the chi2 variables
chi2rz_bins = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0, np.inf])
chi2rphi_bins = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 10.0, 15.0, 20.0, 35.0, 60.0, 200.0, np.inf])
bendchi2_bins = np.array([0.0, 0.75, 1.0, 1.5, 2.25, 3.5, 5.0, 20.0, np.inf])

X_train_bin = X_train.copy()
X_train_bin[:,6] = np.digitize(X_train_bin[:,6]/(X_train_bin[:,4]-2),chi2rphi_bins)-1 #chi2rphi
X_train_bin[:,7] = np.digitize(X_train_bin[:,7]/(X_train_bin[:,4]-2),chi2rz_bins)-1 #chi2rz
X_train_bin[:,3] = np.digitize(X_train_bin[:,3],bendchi2_bins)-1 #bendchi2

X_test_bin = X_test.copy()
X_test_bin[:,6] = np.digitize(X_test_bin[:,6]/(X_test_bin[:,4]-2),chi2rphi_bins)-1 #chi2rphi
X_test_bin[:,7] = np.digitize(X_test_bin[:,7]/(X_test_bin[:,4]-2),chi2rz_bins)-1 #chi2rz
X_test_bin[:,3] = np.digitize(X_test_bin[:,3],bendchi2_bins)-1 #bendchi2

# -----TRAIN MODEL-----

# train gbdt with 60 trees and a max depth of 3
clf_GBDT = xgb.XGBClassifier(n_estimators=60, max_depth=3, random_state=23)
clf_GBDT.fit(X_train_bin,y_train)

# -----EVALUATE MODEL-----

# Create roc curve with AUC (area under curve) value
y_pred = clf_GBDT.predict_proba(X_test_bin)[:,1]
fpr, tpr, dt = roc_curve(y_test,y_pred)
auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr)
plt.xlabel('FPR',fontsize=14)
plt.ylabel('TPR',fontsize=14)
plt.title('AUC = '+str(auc),fontsize=14)
plt.savefig('ROC_curve.png')
plt.close()

# Plot TPR and FPR vs decision threshold (dt)
plt.plot(dt[1:],fpr[1:],label='FPR')
plt.plot(dt[1:],tpr[1:],label='TPR')
plt.xlabel('decision thresh.',fontsize=14)
plt.ylabel('metric',fontsize=14)
plt.legend(loc='best')
plt.savefig('TPR_FPR_vs_dt.png')
plt.close()

# Create TPR/FPR vs. pt
pt, eff, faker, err_eff, err_faker = get_eff_faker_vs_feat('pt',pt_test,X_test_bin,y_test,clf_GBDT)

plt.errorbar(pt,eff,yerr=err_eff,linestyle='None',fmt='.')
plt.xlabel('p$_{T}$ (GeV/c)',fontsize=14)
plt.ylabel('TPR',fontsize=14)
plt.savefig('TPR_vs_pt.png')
plt.close()

plt.errorbar(pt,faker,yerr=err_faker,linestyle='None',fmt='.')
plt.xlabel('p$_{T}$ (GeV/c)',fontsize=14)
plt.ylabel('FPR',fontsize=14)
plt.savefig('FPR_vs_pt.png')
plt.close()

# Create TPR/FPR vs. eta
pt, eff, faker, err_eff, err_faker = get_eff_faker_vs_feat('eta',eta_test,X_test_bin,y_test,clf_GBDT)

plt.errorbar(pt,eff,yerr=err_eff,linestyle='None',fmt='.')
plt.xlabel('$\eta$',fontsize=14)
plt.ylabel('TPR',fontsize=14)
plt.savefig('TPR_vs_eta.png')
plt.close()

plt.errorbar(pt,faker,yerr=err_faker,linestyle='None',fmt='.')
plt.xlabel('$\eta$',fontsize=14)
plt.ylabel('FPR',fontsize=14)
plt.savefig('FPR_vs_eta.png')
plt.close()

# -----SAVE MODEL TO .ONNX-----

# This will output a model.onnx file
initial_type = [('feature_input', FloatTensorType([1, len(data.trks.X_feats)]))]
onx = onnxmltools.convert.convert_xgboost(clf_GBDT, initial_types=initial_type)
with open("default_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
