import os

import torch
import torch.nn as nn

from efficientvit.apps.data_provider import DataProvider, parse_image_size
from efficientvit.apps.trainer.run_config import RunConfig
from efficientvit.apps.utils import EMA
from efficientvit.models.nn.norm import reset_bn
from efficientvit.models.utils import is_parallel, load_state_dict_from_file

__all__ = ["Trainer"]


class Trainer:
    def __init__(self, path: str, model: nn.Module, data_provider: DataProvider):
        self.path = os.path.realpath(os.path.expanduser(path))
        self.model = model.cuda()
        self.data_provider = data_provider

        self.ema = None

        self.checkpoint_path = os.path.join(self.path, "checkpoint")
        self.logs_path = os.path.join(self.path, "logs")
        for path in [self.path, self.checkpoint_path, self.logs_path]:
            os.makedirs(path, exist_ok=True)

        self.best_val = 0.0
        self.start_epoch = 0

    @property
    def network(self) -> nn.Module:
        return self.model.module if is_parallel(self.model) else self.model

    @property
    def eval_network(self) -> nn.Module:
        if self.ema is None:
            model = self.model
        else:
            model = self.ema.shadows
        model = model.module if is_parallel(model) else model
        return model

    def write_log(self, log_str, prefix="valid", print_log=True, mode="a") -> None:
        fout = open(os.path.join(self.logs_path, f"{prefix}.log"), mode)
        fout.write(log_str + "\n")
        fout.flush()
        fout.close()
        if print_log:
            print(log_str)

    def save_model(
        self,
        checkpoint=None,
        only_state_dict=True,
        epoch=0,
        model_name=None,
    ) -> None:

        if checkpoint is None:
            if only_state_dict:
                checkpoint = {"state_dict": self.network.state_dict()}
            else:
                checkpoint = {
                    "state_dict": self.network.state_dict(),
                    "epoch": epoch,
                    "best_val": self.best_val,
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "ema": self.ema.state_dict() if self.ema is not None else None,
                    "scaler": self.scaler.state_dict() if self.fp16 else None,
                }

        model_name = model_name or "checkpoint.pt"

        latest_fname = os.path.join(self.checkpoint_path, "latest.txt")
        model_path = os.path.join(self.checkpoint_path, model_name)
        with open(latest_fname, "w") as _fout:
            _fout.write(model_path + "\n")
        torch.save(checkpoint, model_path)

    def load_model(self, model_fname=None) -> None:
        latest_fname = os.path.join(self.checkpoint_path, "latest.txt")
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, "r") as fin:
                model_fname = fin.readline()
                if len(model_fname) > 0 and model_fname[-1] == "\n":
                    model_fname = model_fname[:-1]
        try:
            if model_fname is None:
                model_fname = f"{self.checkpoint_path}/checkpoint.pt"
            elif not os.path.exists(model_fname):
                model_fname = f"{self.checkpoint_path}/{os.path.basename(model_fname)}"
                if not os.path.exists(model_fname):
                    model_fname = f"{self.checkpoint_path}/checkpoint.pt"
            print(f"=> loading checkpoint {model_fname}")
            checkpoint = load_state_dict_from_file(model_fname, False)
        except Exception:
            self.write_log(f"fail to load checkpoint from {self.checkpoint_path}")
            return

        # load checkpoint
        self.network.load_state_dict(checkpoint["state_dict"], strict=False)
        log = []
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
            self.run_config.update_global_step(self.start_epoch)
            log.append(f"epoch={self.start_epoch - 1}")
        if "best_val" in checkpoint:
            self.best_val = checkpoint["best_val"]
            log.append(f"best_val={self.best_val:.2f}")
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            log.append("optimizer")
        if "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            log.append("lr_scheduler")
        if "ema" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
            log.append("ema")
        if "scaler" in checkpoint and self.fp16:
            self.scaler.load_state_dict(checkpoint["scaler"])
            log.append("scaler")
        self.write_log("Loaded: " + ", ".join(log))

   
    """ validate """


    def _validate(self, model, data_loader, epoch) -> dict[str, any]:
        raise NotImplementedError

    def validate(self, model=None, data_loader=None, is_test=True, epoch=0) -> dict[str, any]:
        model = model or self.eval_network
        if data_loader is None:
            if is_test:
                data_loader = self.data_provider.test
            else:
                data_loader = self.data_provider.valid

        model.eval()
        return self._validate(model, data_loader, epoch)


    """ training """

    def prep_for_training(self, run_config: RunConfig, ema_decay: float or None = None, fp16=False) -> None:
        self.run_config = run_config
        #self.model = nn.DataParallel(self.model) # Parallel GPU
        self.model.cuda()

        self.run_config.global_step = 0
        self.run_config.batch_per_epoch = len(self.data_provider.train)
        assert self.run_config.batch_per_epoch > 0, "Training set is empty"

        # build optimizer
        self.optimizer, self.lr_scheduler = self.run_config.build_optimizer(self.model)


        # fp16
        self.fp16 = fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

    def sync_model(self):
        print("Sync model")
        self.save_model(model_name="sync.pt")
        dist.barrier()
        checkpoint = torch.load(os.path.join(self.checkpoint_path, "sync.pt"), map_location="cpu")
        dist.barrier()
        if dist.is_master():
            os.remove(os.path.join(self.checkpoint_path, "sync.pt"))
        dist.barrier()

        # load checkpoint
        self.network.load_state_dict(checkpoint["state_dict"], strict=False)
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "ema" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        if "scaler" in checkpoint and self.fp16:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def before_step(self, feed_dict: dict[str, any]) -> dict[str, any]:
        for key in feed_dict:
            if isinstance(feed_dict[key], torch.Tensor):
                feed_dict[key] = feed_dict[key].cuda()
        return feed_dict

    def run_step(self, feed_dict: dict[str, any]) -> dict[str, any]:
        raise NotImplementedError

    def after_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        # gradient clip
        if self.run_config.grad_clip is not None:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.run_config.grad_clip)
        # update
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.lr_scheduler.step()
        self.run_config.step()
        # update ema
        if self.ema is not None:
            self.ema.step(self.network, self.run_config.global_step)

    def _train_one_epoch(self, epoch: int) -> dict[str, any]:
        raise NotImplementedError

    def train_one_epoch(self, epoch: int) -> dict[str, any]:
        self.model.train()

        self.data_provider.set_epoch(epoch)

        train_info_dict = self._train_one_epoch(epoch)

        return train_info_dict

    def train(self) -> None:
        raise NotImplementedError
