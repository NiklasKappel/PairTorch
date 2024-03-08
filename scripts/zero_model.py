import torch


class ZeroModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, types, positions, edge_index, batch: torch.Tensor | None = None):
        del types, edge_index, batch
        energy = torch.zeros(1, 1)
        forces = torch.zeros_like(positions)
        return energy, forces


if __name__ == "__main__":
    model = ZeroModel()
    compiled_model = torch.jit.script(model)
    compiled_model.save("zero_model.pt")  # type: ignore
