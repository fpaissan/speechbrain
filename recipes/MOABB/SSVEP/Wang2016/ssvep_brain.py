from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, accuracy_score
from speechbrain.core import Stage
from torch.nn import init
from torch import Tensor
import speechbrain as sb
import torch

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

class Wang2016Brain(sb.Brain):
    """Brain to solve SSVEP on Wang2016 benchmark"""
    def compute_forward(self, 
                        batch: Tensor,  
                        stage: Stage) \
                            -> Tensor:
        """Given an input batch, computes forward step
        on defined model.

        :param batch: batch of data from dataloader
        :type batch: Tensor
        :param stage: stage, to track exp stage
        :type stage: Stage
        :return: output of NN model
        :rtype: Tensor
        """
        inputs = batch[0].to(self.device)
        inputs = inputs.permute(0, 3, 1, 2)
        input(self.modules.model)
        out = self.modules.model(inputs)
        input(out.shape)
        return out

    def compute_objectives(self, 
                           predictions: Tensor, 
                           batch: Tensor, 
                           stage: Stage) \
                               -> Tensor:
        """Given the network predictions and targets computes the loss.

        :param predictions: NN predictions
        :type predictions: Tensor
        :param batch: batch of data from dataloader
        :type batch: Tensor
        :param stage: stage, to track exp stage
        :type stage: Stage
        :return: loss function, as defined in YAML file
        :rtype: Tensor
        """
        targets = batch[1].to(self.device)
        # input(f"{predictions.shape} - {targets.shape}")
        loss = self.hparams.loss(
            predictions,
            targets,
            weight=torch.FloatTensor(self.hparams.class_weight).to(self.device),
        )
        if stage != sb.Stage.TRAIN:
            tmp_preds = torch.exp(predictions)
            self.preds.extend(tmp_preds.detach().cpu().numpy())
            self.targets.extend(batch[1].detach().cpu().numpy())
        return loss

    def on_fit_start(self) -> None:
        """Gets called at the beginning of ``fit()``"""
        initialize_module(self.hparams.model)
        self.init_optimizers()
        self.metrics = {}
        self.metrics["loss"] = []
        self.metrics["f1"] = []
        self.metrics["auc"] = []
        self.metrics["cm"] = []
        self.metrics["acc"] = []

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage != sb.Stage.TRAIN:
            self.preds = []
            self.targets = []

    def on_stage_end(self, 
                     stage: Stage, 
                     stage_loss: float, 
                     epoch: int = None) \
                         -> None:
        """Gets called at the end of an epoch.

        :param stage: stage, to track exp stage
        :type stage: Stage
        :param stage_loss: average loss over the completed stage
        :type stage_loss: float
        :param epoch: epoch count, defaults to None
        :type epoch: int, optional
        """
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            preds = np.array(self.preds)
            y_pred = np.argmax(preds, axis=-1)
            y_true = self.targets
            f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
            auc = roc_auc_score(
                y_true=y_true, y_score=preds, multi_class="ovo", average="macro"
            )
            acc = accuracy_score(
                y_true=y_true,
                y_pred=y_pred
            )
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
            self.last_eval_loss = stage_loss
            self.last_eval_f1 = float(f1)
            self.last_eval_auc = float(auc)
            self.last_eval_cm = cm
            self.last_eval_acc = float(acc)
        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": self.last_eval_loss,
                    "f1": self.last_eval_f1,
                    "auc": self.last_eval_auc,
                    "cm": self.last_eval_cm,
                    "acc": self.last_eval_acc
                },
            )
            # track valid metric history
            self.metrics["loss"].append(self.last_eval_loss)
            self.metrics["f1"].append(self.last_eval_f1)
            self.metrics["auc"].append(self.last_eval_auc)
            self.metrics["cm"].append(self.last_eval_cm)
            self.metrics["acc"].append(self.last_eval_acc)
            min_key, max_key = None, None
            if self.hparams.direction == "max":
                min_key = None
                max_key = self.hparams.target_valid_metric
            elif self.hparams.direction == "min":
                min_key = self.hparams.target_valid_metric
                max_key = None

            self.checkpointer.save_and_keep_only(
                meta={
                    "loss": self.metrics["loss"][-1],
                    "f1": self.metrics["f1"][-1],
                    "auc": self.metrics["auc"][-1],
                },
                min_keys=[min_key],
                max_keys=[max_key],
            )

            # early stopping
            current_metric = self.metrics[self.hparams.target_valid_metric][-1]
            if self.hparams.epoch_counter.should_stop(
                current=epoch, current_metric=current_metric,
            ):
                self.hparams.epoch_counter.current = (
                    self.hparams.epoch_counter.limit
                )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch loaded": self.hparams.epoch_counter.current},
                test_stats={
                    "loss": self.last_eval_loss,
                    "f1": self.last_eval_f1,
                    "auc": self.last_eval_auc,
                    "cm": self.last_eval_cm,
                    "acc": self.last_eval_acc
                },
            )
