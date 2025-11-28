import torch
from torch.utils.data import DataLoader

Tensor = torch.Tensor
Input = Tensor | DataLoader
RolloutOutput = tuple[Tensor, None] | tuple[Tensor, Tensor]

Batch = dict[str, Tensor]

# TODO: Could be a dataclass if we want more structure
# @dataclass
# class Batch:
#     input_fields: Tensor
#     output_fields: Tensor
#     constant_scalars: Tensor
#     constant_fields: Tensor
