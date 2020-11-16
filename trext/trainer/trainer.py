from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils as utils
import tqdm
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
            self,
            logger,
            max_epoch: int,
            verbose: bool,
            version: str,
        ):
        self.logger = logger
        self.max_epoch = max_epoch
        self.verbose = verbose
        self.version = version

    def save_checkpoint(
            self,
            model: Module,
            optimizer: Optimizer,
            epoch_idx: int,
            checkpoints_dir: Path,
        ) -> None:
        checkpoint = {
            'model': model,
            'optimizer': optimizer,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch_idx': epoch_idx,
        }
        checkpoint_path = checkpoints_dir / f"v{self.version}-e{epoch_idx}.hdf5"
        torch.save(checkpoint, checkpoint_path)

    @torch.enable_grad()
    def training_epoch(
            self,
            model: Module,
            train_dataloader: DataLoader,
            optimizer: Optimizer,
            epoch_idx: int,
        ) -> None:
        model.train()

        for batch_idx, batch in enumerate(tqdm.tqdm(train_dataloader)):
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
            optimizer.step()
            optimizer.zero_grad()
            model.training_step_end()

        model.training_epoch_end(epoch_idx=epoch_idx)

    @torch.no_grad()
    def validation_epoch(
            self,
            model: Module,
            val_dataloader: DataLoader,
            scheduler: _LRScheduler,
            epoch_idx: int,
        ) -> None:
        model.eval()
        loss_sum = 0

        for batch_idx, batch in enumerate(tqdm.tqdm(val_dataloader)):
            loss = model.validation_step(batch, batch_idx)
            loss_sum += loss.item()
            model.validation_step_end()

        #scheduler.step(loss_sum)

        print(epoch_idx, loss_sum / len(val_dataloader))
        model.validation_epoch_end(epoch_idx=epoch_idx)

    def fit(
            self,
            model: Module,
            datamodule,
        ) -> None:
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        optimizer, scheduler = model.configure_optimizers()

        def init_weights(m):
            if hasattr(m, 'weight'):
                if m.weight is None:
                    pass
                elif m.weight.dim() > 1:
                    nn.init.xavier_uniform_(m.weight.data)

        model.apply(init_weights)

        self.validation_epoch(
            model=model,
            val_dataloader=val_dataloader,
            scheduler=scheduler,
            epoch_idx=0,
        )
        for epoch_idx in range(1, self.max_epoch + 1):
            self.training_epoch(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                epoch_idx=epoch_idx,
            )
            self.validation_epoch(
                model=model,
                val_dataloader=val_dataloader,
                scheduler=scheduler,
                epoch_idx=epoch_idx,
            )

            if epoch_idx % 2 == 0:
                self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch_idx=epoch_idx,
                    checkpoints_dir=Path.cwd() / "models",
                )

        return model

    @torch.no_grad()
    def predict(
            self,
            model: Module,
            datamodule,
        ):
        test_dataloader = datamodule.test_dataloader()

        predicts = list()

        for batch_idx, batch in enumerate(test_dataloader):
            predict = model.test_step(batch, batch_idx)
            predicts.append(predict)
            model.test_step_end()

        model.test_epoch_end()

        return predicts

