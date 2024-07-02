import torch
import util
import pytest

v1 = torch.tensor([1.0, 2.0, 3.0])
print(util.addimag(v1))