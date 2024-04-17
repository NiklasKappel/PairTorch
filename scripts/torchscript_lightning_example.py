# pyright: reportIncompatibleMethodOverride=false

from typing import List, Optional

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.rand(1000, 3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MyDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=7):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        del stage  # unused
        self.dataset = MyDataset()

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class MyModel(L.LightningModule):
    def __init__(self, input_size: int):
        super().__init__()
        # Save hyperparameters so that the model can be loaded from a checkpoint.
        self.save_hyperparameters()
        self.layer = torch.nn.Linear(input_size, 1)

    def forward(self, positions: Tensor):
        positions.requires_grad_(True)

        energy = self.layer(positions)

        # The signatures of grad() in Python and C++ do not match.
        # To make the force calculation work in TorchScript,
        # we need to supply the arguments to grad() exactly as below,
        # and assert that the force output is not None.
        grad_outputs: List[Optional[Tensor]] = [torch.ones_like(energy)]
        forces = torch.autograd.grad(
            [energy],
            [positions],
            grad_outputs=grad_outputs,  # type: ignore
            create_graph=True,
        )[0]
        assert forces is not None
        forces = -forces

        return energy, forces

    def training_step(self, batch):
        return self._common_step(batch)

    # Lightning by default disables gradient computation in validation and test steps.
    # To re-enable the force calculation, we need to pass inference_mode=False to the Trainer,
    # and wrap the code in enable_grad(). However, enable_grad() may not be applied to forward(),
    # because that is not supported by TorchScript.
    @torch.enable_grad()
    def validation_step(self, batch):
        return self._common_step(batch)

    @torch.enable_grad()
    def test_step(self, batch):
        return self._common_step(batch)

    def _common_step(self, batch):
        energy, forces = self(batch)
        loss = energy.mean() + forces.mean()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def main():
    # Train and save checkpoint.
    trainer = L.Trainer(enable_checkpointing=False, inference_mode=False, max_epochs=1)
    model = MyModel(input_size=3)
    dataset = MyDataModule()
    trainer.fit(model, dataset)
    trainer.test(model, dataset)
    trainer.save_checkpoint("model.ckpt")

    # Load checkpoint and save TorchScript.
    model_2 = MyModel.load_from_checkpoint("model.ckpt")
    script = model_2.to_torchscript()
    torch.jit.save(script, "model.pt")

    # Load and run TorchScript.
    script_2 = torch.jit.load("model.pt")
    print(script_2.state_dict())
    input = torch.rand(2, 3)
    print(input)
    energy, forces = script_2(input)
    print(energy)
    print(forces)
    print(script_2.code)


if __name__ == "__main__":
    main()
