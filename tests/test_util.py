import torch
import pytest
from functions.util import addimag

def test_addimag():
    # Test case 1
    v1 = torch.tensor([1.0, 2.0, 3.0])
    expected_output1 = torch.tensor([1.0, 0.0, -1.0])
    assert torch.allclose(addimag(v1), expected_output1)

    # Test case 2
    v2 = torch.tensor([0.5, 0.5, 0.5])
    expected_output2 = torch.tensor([0.5, 0.0, -0.5])
    assert torch.allclose(addimag(v2), expected_output2)

    # Test case 3
    v3 = torch.tensor([0.0, 0.0, 0.0])
    expected_output3 = torch.tensor([0.0, 0.0, 0.0])
    assert torch.allclose(addimag(v3), expected_output3)

    # Test case 4
    v4 = torch.tensor([1.0, -1.0, 1.0])
    expected_output4 = torch.tensor([1.0, 0.0, -1.0])
    assert torch.allclose(addimag(v4), expected_output4)

    # Test case 5
    v5 = torch.tensor([2.0, 4.0, 6.0])
    expected_output5 = torch.tensor([2.0, 0.0, -2.0])
    assert torch.allclose(addimag(v5), expected_output5)

    print("All test cases pass")

test_addimag()