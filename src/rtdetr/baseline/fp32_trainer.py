"""RT-DETR trainer that keeps checkpoints in fp32.

Ultralytics' BaseTrainer.save_model stores the EMA model with .half(), and
final_eval's strip_optimizer converts last.pt/best.pt to fp16 again.
RT-DETR's BatchNorm running_var exceeds fp16's maximum (65504) in a growing
number of channels from epoch 4 on; the overflow becomes inf and is
clamped to 65504 by torch.nan_to_num_, so every saved checkpoint
normalises those channels wrongly (results/rtdetr/baseline_first_run:
mAP50 0.51 during training, 0 when best.pt/last.pt are reloaded).

Fp32RTDETRTrainer is RTDETRTrainer with those two methods copied from
ultralytics 8.4.77 (engine/trainer.py) and changed only there:
  * save_model: no .half() on the saved EMA copy;
  * final_eval: no strip_optimizer on last.pt / best.pt.
Checkpoints are about twice as large.

A separate module, not a class inside train.py: on several GPUs Ultralytics
re-launches training in a generated script that imports the trainer class
by module name (train.sbatch puts this folder on PYTHONPATH).
"""

import io  # noqa: F401  (save_model imports it locally as well)

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.engine.trainer import (  # names the copied methods use
    GIT, LOCAL_RANK, LOGGER, RANK, __version__, convert_optimizer_state_dict_to_fp16,
    datetime, deepcopy, torch, torch_distributed_zero_first, unwrap_model,
)


class Fp32RTDETRTrainer(RTDETRTrainer):
    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Skip only genuinely non-finite fp32 EMA weights; fp16 overflow is handled on the saved copy below.
        ema = unwrap_model(self.ema.ema)
        if not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor)):
            LOGGER.warning(f"Skipping checkpoint save at epoch {self.epoch}: EMA contains NaN/Inf")
            return False
        ema = deepcopy(ema)  # fp32: .half() clipped BatchNorm running_var at 65504
        # Clamp fp16 serialization overflow without mutating the live EMA.
        for v in ema.state_dict().values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                torch.nan_to_num_(v)

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": ema,
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "message": GIT.message,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.wdir.mkdir(parents=True, exist_ok=True)  # ensure weights directory exists
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
        return True

    def final_eval(self):
        """Perform final evaluation and validation for the YOLO model."""
        model = self.best if self.best.exists() else None
        # No strip_optimizer: it converts last.pt/best.pt to fp16 (clipping again).
        if model:
            LOGGER.info(f"\nValidating {model}...")
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False  # disable final val compile as too slow
            self.metrics = self.validator(model=model)
            self.metrics.pop("fitness", None)
            self.epoch += 1  # log best metrics at step epochs+1, not overwriting last epoch
            self.run_callbacks("on_fit_epoch_end")
            self.epoch -= 1  # restore epoch
