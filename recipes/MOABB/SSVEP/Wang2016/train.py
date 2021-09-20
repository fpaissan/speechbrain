#!/usr/bin/python
"""
Recipe for training a compact CNN to decode steady state visually evoked potential (SSVEP) in EEG trials.
The CNN is based on EEGNet and the dataset is Wang2016 from MOABB.
Reference to EEGNet: V. J. Lawhern et al., J Neural Eng 2018 (https://doi.org/10.1088/1741-2552/aace8c).
Reference to Wang2016:  (https://ieeexplore.ieee.org/document/7740878).

To run this recipe:

    > TODO
    # > python3 train.py train.yaml --data_folder '/path/to/MOABB_BNCI2014001'

Author
------
Francesco Paissan, 2021
"""
from MOABB_dataio_iterators import WithinSession, CrossSession, LeaveOneSubjectOut
from hyperpyyaml import load_hyperpyyaml
from moabb.datasets import Wang2016
from moabb.paradigms import SSVEP
from ssvep_utils import run_single_fold
import os
import pickle
import speechbrain as sb
import sys

if __name__ == "__main__":
    argv = sys.argv[1:]
    # Temporary switching off deprecation warning from mne
    import warnings  # noqa

    warnings.filterwarnings("ignore")
    hparams_file, run_opts, overrides = sb.core.parse_arguments(argv)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    hparams["class_weight"] = hparams["class_weight"]*hparams["num_classes"] 

    moabb_dataset = Wang2016()
    moabb_dataset.download(path=hparams["data_folder"])
    moabb_paradigm = SSVEP(
        n_classes=len(hparams["class_weight"]),
        fmin=hparams["fmin"],
        fmax=hparams["fmax"],
        tmin=hparams["tmin"],
        tmax=hparams["tmax"],
        resample=hparams["sf"],
    )
    
    
    # defining data iterators to use
    data_its = [
        WithinSession(moabb_paradigm, hparams),
        # CrossSession(moabb_paradigm, hparams),
        # LeaveOneSubjectOut(moabb_paradigm, hparams),
    ]
    for data_it in data_its:
        print("Running {0} iterations".format(data_it.iterator_tag))
        for i, (exp_dir, datasets) in enumerate(data_it.prepare(moabb_dataset)):
            print("Running experiment %i" % (i))
            hparams["exp_dir"] = exp_dir
            # creating experiment directory
            sb.create_experiment_directory(
                experiment_directory=hparams["exp_dir"],
                hyperparams_to_save=hparams_file,
                overrides=overrides,
            )
            tmp_metrics_dict = run_single_fold(hparams, run_opts, datasets)
            # saving metrics on the test set in a pickle file
            metrics_fpath = os.path.join(hparams["exp_dir"], "metrics.pkl")
            with open(metrics_fpath, "wb",) as handle:
                pickle.dump(
                    tmp_metrics_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
                )
            # restoring hparams for the next training and evaluation processes
            hparams_file, run_opts, overrides = sb.core.parse_arguments(argv)
            with open(hparams_file) as fin:
                hparams = load_hyperpyyaml(fin, overrides)
