from ssvep_brain import Wang2016Brain
from torch.nn import init

def run_single_fold(hparams: dict, run_opts: dict, datasets: dict) -> dict:
    """Performs training and evaluation on single a fold.

    :param hparams: hyper params for training
    :type hparams: dict
    :param run_opts: run options, as required by sb.Brain
    :type run_opts: dict
    :param datasets: dictionary contaning dataset for stages
    :type datasets: dict
    :return: [description]
    :rtype: dict
    """
    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=os.path.join(hparams["exp_dir"], "save"),
        recoverables={
            "model": hparams["model"],
            "counter": hparams["epoch_counter"],
        },
    )
    
    hparams["train_logger"] = sb.utils.train_logger.FileTrainLogger(
        save_file=os.path.join(hparams["exp_dir"], "train_log.txt")
    )
    
    brain = Wang2016Brain(
        modules={"model": hparams["model"]},
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=checkpointer,
    )
    
    # training
    brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        progressbar=True,
    )
    
    # evaluation
    min_key, max_key = None, None
    if hparams["direction"] == "max":
        min_key = None
        max_key = hparams["target_valid_metric"]
        
    elif hparams["direction"] == "min":
        min_key = hparams["target_valid_metric"]
        max_key = None

    brain.evaluate(
        datasets["test"], progressbar=False, min_key=min_key, max_key=max_key
    )
    
    test_loss, test_f1, test_auc, test_cm, test_acc = (
        brain.last_eval_loss,
        brain.last_eval_f1,
        brain.last_eval_auc,
        brain.last_eval_cm,
        brain.last_eval_acc
    )

    tmp_metrics_dict = {
        "loss": test_loss,
        "f1": test_f1,
        "auc": test_auc,
        "cm": test_cm,
        "acc": test_acc
    }
    
    return tmp_metrics_dict

def initialize_module(module):
    """Function to initialize neural network modules"""
    for mod in module.modules():
        if hasattr(mod, "weight"):
            if not ("BatchNorm" in mod.__class__.__name__):
                init.xavier_uniform_(mod.weight, gain=1)
            else:
                init.constant_(mod.weight, 1)
        if hasattr(mod, "bias"):
            if mod.bias is not None:
                init.constant_(mod.bias, 0)