# TrainTrackQuality/DefaultModel

This directory contains code that will recreate the default L1 track trigger track quality [model](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/TrackTrigger/python/TrackQualityParams_cfi.py#L4) found in CMSSW. The model may vary slighlty due to the randomness in training and differences in the ntuple input. More information about the default model can be found [here](https://indico.cern.ch/event/974811/contributions/4104918/attachments/2144974/3615326/L1T_11_17_20.pdf).

## Dependencies

To recreate the default model, you must run a Z->ee, Z->uu, and QCD +200PU sample through the default [ntuplemaker](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/TrackFindingTracklet/test/L1TrackNtupleMaker_cfg.py) found in L1Trigger/TrackFindingTracklet/test. Combine the 3 separate samples into one (using `hadd ...` for example) and use this as input to the `create_default_gbdt.py` script (line 15).

There are also a few packages that need to be installed:
- [uproot](https://pypi.org/project/uproot/) used to import .root files
- [Scikit-Learn](https://scikit-learn.org/stable/install.html) used to evaluate all models
- [NumPy](https://numpy.org/install/) used for many array and mathematical operations
- [onnxmltools](https://pypi.org/project/onnxmltools/1.0.0.0/) used to convert models to onnx
- [matplotlib](https://matplotlib.org/stable/users/installing.html) used to create the graphs for evaluation
- [XGBoost](https://xgboost.readthedocs.io/en/latest/) used to create and train the model

## To run

To run an example script, do `python create_default_gbdt.py` and the outputs will be `.png` files that evaluate the model along with a `default_model.onnx` which is the final model format that can be imported into CMSSW.