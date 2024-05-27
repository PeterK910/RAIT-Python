import torch
"""
Derivatives of the argument function of a Blaschke product.

:param a: parameters of the Blaschke product, one-dimensional tensor with complex numbers
:type a: tensor
:param t: values in [-pi,pi), where the function values are needed, one-dimensional tensor with floats
:type t: tensor

:returns: the derivatives of the argument function at the points in "t", one-dimensional tensor with floats
:rtype: tensor
"""
def arg_der(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # Convert inputs to PyTorch tensors if they are not already
    a = torch.tensor(a, dtype=torch.complex64)
    t = torch.tensor(t, dtype=torch.float32)
    
    # Check if inputs are 1D tensors
    if a.dim() != 1 or t.dim() != 1:
        raise ValueError('Parameters should be 1D tensors!')
    if torch.max(torch.abs(a)) >= 1:
        raise ValueError('Bad poles!')
    
    bd = torch.zeros(t.size())
    for i in range(a.size(0)):
        bd = bd + __arg_der_one(a[i], t)
    bd = bd / a.size(0)
    return bd
"""
calculate the derivative at each point in t
"""
def __arg_der_one(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r = torch.abs(a)
    fi = torch.angle(a)
    
    bd = (1 - r**2) / (1 + r**2 - 2*r*torch.cos(t-fi))
    return bd
"""
 Gives a sampled Blaschke function.

:param length: number of points to sample
:type length: int
:param poles: poles of the Blaschke product, one-dimensional tensor with complex numbers
:type poles: tensor

:returns: the Blaschke product with given parameters sampled at uniform points on the torus, one-dimensional tensor with complex numbers
:rtype: tensor
"""
def blaschkes(length:int, poles:torch.Tensor) -> torch.Tensor:
    # Convert inputs to PyTorch tensors if they are not already
    poles = torch.tensor(poles, dtype=torch.complex64)
    
    # Check if poles is a 1D tensor
    if poles.dim() != 1:
        raise ValueError('Poles should be a 1D tensor!')
    
    # Create a tensor of linearly spaced values from -pi to pi
    t = torch.linspace(-torch.pi, torch.pi, length + 1)
    z = torch.exp(1j * t)
    b = torch.ones(length + 1, dtype=torch.complex64)
    
    # Compute the Blaschke product for each pole
    for pole in poles:
        b = b * (z - pole) / (1 - torch.conj(pole) * z)
    
    return b

"""
Values of the argument function of a Blaschke product.

:param a: parameters of the Blaschke product, one-dimensional tensor with complex numbers
:type a: tensor
:param t: values in [-pi,pi), where the function values are needed, one-dimensional tensor with floats
:type t: tensor

:returns: the values of the argument function at the points in "t", one-dimensional tensor with floats
"""
def arg_fun(a:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
    if a.ndim != 1 or t.ndim != 1:
        raise ValueError('Parameters should be 1D tensors!')
    if torch.max(torch.abs(a)) >= 1:
        raise ValueError('Bad poles!')

    b = torch.zeros_like(t)
    for i in range(a.size(0)):
        b += __arg_fun_one(a[i], t)
    b /= a.size(0)
    return b
"""
calculate the argument at a given point in t
"""
def __arg_fun_one(a:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
    r = torch.abs(a)
    fi = torch.angle(a)
    mu = (1 + r) / (1 - r)
    gamma = 2 * torch.atan((1 / mu) * torch.tan(fi / 2))
    b = 2 * torch.atan(mu * torch.tan((t - fi) / 2)) + gamma
    b = torch.fmod(b + torch.pi, 2 * torch.pi) - torch.pi  # move it in [-pi,pi)
    return b
"""
Same function as 'arg_fun', but it is continuous on IR. 
TODO: types of arguments and return value (what type of numbers do the tensors contain?)
"""
def argdr_fun(a:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
    b = torch.zeros(t.size())
    for j in range(t.numel()):
        for i in range(a.numel()):
            bs = arg_fun(a[i], t[j])
            b[j] += bs + 2 * torch.pi * torch.floor((t[j] + torch.pi) / (2 * torch.pi))
    return b