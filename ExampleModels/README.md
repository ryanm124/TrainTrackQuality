# TrainTrackQuality/ExampleModels

This directory contains python scripts that create example machine learning track quality classifiers for L1TK reconstructed tracks. 

Their are examples of how to create 2 different classifiers using 3 different machine learning packages:  XGBoost and Scikit-Learn for a gradient boosted decision tree (gbdt), and Keras for a neural network (nn). Each example trains and evaluates the model, then saves the model in the proper .onnx format that can be used in CMSSW as the [L1TK track quality model](https://github.com/cms-L1TK/cmssw/blob/L1TK-dev-11_2_0_pre6/L1Trigger/TrackTrigger/python/TrackQualityParams_cfi.py#L4).

There is also a `functions.py` script which provides the users with a few functions that help create training and testing sets, along with functions to help evaluate models.

## Dependencies

Each example takes in a `*.root` ntuple which holds all of the data for training and testing the model. This ntuple can be created from the default [L1NtupleMaker](https://github.com/cms-L1TK/cmssw/blob/L1TK-dev-11_2_0_pre6/L1Trigger/TrackFindingTracklet/test/L1TrackNtupleMaker_cfg.py) in the L1TK/TrackFindingTracklet/test folder. You will have to update the name of the ntuple in the example scripts before you run.

To run any examples, you must first import these packages:
- [uproot](https://pypi.org/project/uproot/) used to import .root files
- [Scikit-Learn](https://scikit-learn.org/stable/install.html) used to evaluate all models
- [NumPy](https://numpy.org/install/) used for many array and mathematical operations
- [onnxmltools](https://pypi.org/project/onnxmltools/1.0.0.0/) used to convert models to onnx
- [matplotlib](https://matplotlib.org/stable/users/installing.html) used to create the graphs for evaluation

In addition to the above packages, each `example_*.py` script requires a few additional python packages to run. 
These packages are indicated below:
- `example_xgboost_gbdt.py`: [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- `example_scikitlearn_gbdt.py`: nothing in addition to the packages already mentioned above
- `example_keras_nn.py`: [Keras](https://keras.io/)

## To run

To run an example script, do `python example_<model>.py` and the outputs will be `.png` files that evaluate the model along with a `model.onnx` which is the final model format that can be imported into CMSSW.
