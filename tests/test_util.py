import torch
import pytest
import re 

def test_check_poles():
    from .util import check_poles
    #a
    regex = re.compile(re.escape('poles must be a torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        poles = [1, 2, 3]
        check_poles(poles)

    regex = re.compile(re.escape('poles must be complex numbers.'))
    with pytest.raises(TypeError, match=regex):
        poles = torch.tensor([1.0, 2.0, 3.0])
        check_poles(poles)

    regex = re.compile(re.escape('poles must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        poles = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.complex64)
        check_poles(poles)

    regex = re.compile(re.escape('poles must be inside the unit circle!'))
    with pytest.raises(ValueError, match=regex):
        poles = torch.tensor([0.5, 0.5, 1], dtype=torch.complex64)
        check_poles(poles)

#addimag test
def test_addimag():
    from .util import addimag
    v1 = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(addimag(v1), torch.tensor([2.+0.j, 2.+0.j, 2.+0.j]))
    v2 = torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert torch.equal(addimag(v2), torch.tensor([1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]))
    v3 = torch.tensor([-1.0, 0.0, 1.0, 4.0])
    assert torch.equal(addimag(v3), torch.tensor([ 0.+2.j, -1.-1.j,  2.-2.j,  3.+1.j]))

    #input validation
    with pytest.raises(TypeError, match="v must be a torch.Tensor."):
        v4 = [1, 2, 3]
        addimag(v4)
    with pytest.raises(TypeError, match="v must have real elements."):
        v5 = torch.tensor([1+0j, 2, 3])
        addimag(v5)
    with pytest.raises(ValueError, match="v must be a 1-dimensional torch.Tensor."):
        v6 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        addimag(v6)

def test_bisection_order():
    from .util import bisection_order
    result = bisection_order(4)
    expected_result = torch.tensor(
        [[ 0, -1, -1],
        [ 4, -1, -1],
        [ 2,  0,  4],
        [ 1,  0,  2],
        [ 3,  2,  4]], dtype=torch.int32)
    assert torch.equal(result, expected_result)
    #input validation
    with pytest.raises(ValueError, match="n must be a non-negative integer."):
        bisection_order(5.3)
    with pytest.raises(ValueError, match="n must be a non-negative integer."):
        bisection_order(-1)

def test_multiplicity():
    from .util import multiplicity

    a = torch.tensor([0,0,0.5,0,-0.5j], dtype=torch.complex64)
    spoles, mult = multiplicity(a)
    expected_spoles = torch.tensor([0, 0.5, -0.5j], dtype=torch.complex64)
    expected_mult = torch.tensor([3, 1, 1], dtype=torch.int32)
    assert torch.allclose(spoles, expected_spoles)
    assert torch.equal(mult, expected_mult)

def test_kernel():
    from .util import kernel
    #TODO: type specification of y and z
    #when y and z are equal
    y=torch.tensor([0,1], dtype=torch.complex64)
    z=torch.tensor([0,1], dtype=torch.complex64)
    mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_result = torch.tensor([4.6000+0.j, 4.6000+0.j],dtype=torch.complex64)
    assert torch.allclose(kernel(y, z, mpoles), expected_result)

    #when y and z are different
    y=torch.tensor([0,1], dtype=torch.complex64)
    z=torch.tensor([0,-1], dtype=torch.complex64) #-1 instead of 1
    mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_result = torch.tensor([1.0000+0.0000j, 0.3600+0.4800j],dtype=torch.complex64)

def test_discretize_dc():
    from .util import discretize_dc
    #waiting for blaschke/arg_inv to be tested