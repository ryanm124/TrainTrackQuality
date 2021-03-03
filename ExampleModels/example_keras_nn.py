import os
os.environ['TF_KERAS'] = '1'

import uproot3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.backend import clear_session
from functions import get_eff_faker_err, train_test_split_by_part, get_eff_faker_vs_feat
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

# -----IMPORT .ROOT FILES-----

# Take in .root into arrays
arrays = (uproot3.open("TTbar_PU200_D49_prompt.root")["L1TrackNtuple/eventTree"]
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

# This creates NN with 2 hidden layers and one output layer
# The first hidden layer has 128 nodes, and the second has 64 nodes
# The output is a single node with a value between 0 and 1
# Visit https://keras.io/ for more information on how to configure NN
clf_NN = Sequential()
clf_NN.add(Dense(128, input_dim=len(features), activation='relu')) # each Dense layer is a fully-connected layer in the NN being added
clf_NN.add(Dense(64, activation='relu'))
clf_NN.add(Dense(1, activation='sigmoid'))
clf_NN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
clf_NN.fit(X_train, y_train, epochs=100, batch_size=64, verbose=0) #verbose is supressing output

# -----EVALUATE MODEL-----

# Create roc curve with AUC (area under curve) value
y_pred = clf_NN.predict(X_test)[:,0]
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
pt, eff, faker, err_eff, err_faker = get_eff_faker_vs_feat('pt',features,X_test,y_test,clf_NN)

plt.errorbar(pt,eff,yerr=err_eff,linestyle='None',fmt='.')
plt.xlabel('p$_{T}$ (GeV/c)',fontsize=14)
plt.ylabel('TPR',fontsize=14)
plt.savefig('TPR_vs_pt.png')

plt.errorbar(pt,faker,yerr=err_faker,linestyle='None',fmt='.')
plt.xlabel('p$_{T}$ (GeV/c)',fontsize=14)
plt.ylabel('FPR',fontsize=14)
plt.savefig('FPR_vs_pt.png')

# Create TPR/FPR vs. eta
pt, eff, faker, err_eff, err_faker = get_eff_faker_vs_feat('eta',features,X_test,y_test,clf_NN)

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
onx = onnxmltools.convert.convert_keras(clf_NN)
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())