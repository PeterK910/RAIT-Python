import torch
import other
"""
Derivatives of the argument function of a Blaschke product.

:param a: parameters of the Blaschke product, one-dimensional Tensor with complex numbers
:type a: Tensor
:param t: values in [-pi,pi), where the function values are needed, one-dimensional Tensor with floats
:type t: Tensor

:returns: the derivatives of the argument function at the points in "t", one-dimensional Tensor with floats
:rtype: Tensor
"""
def arg_der(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # Convert inputs to PyTorch tensors if they are not already
    a.to(dtype=torch.complex64)
    t.to(dtype=torch.float32)
    
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
:param poles: poles of the Blaschke product, one-dimensional Tensor with complex numbers
:type poles: Tensor

:returns: the Blaschke product with given parameters sampled at uniform points on the torus, one-dimensional Tensor with complex numbers
:rtype: Tensor
"""
def blaschkes(length:int, poles:torch.Tensor) -> torch.Tensor:
    # Convert inputs to PyTorch tensors if they are not already
    poles.to(dtype=torch.complex64)
    
    # Check if poles is a 1D Tensor
    if poles.dim() != 1:
        raise ValueError('Poles should be a 1D Tensor!')
    
    # Create a Tensor of linearly spaced values from -pi to pi
    t = torch.linspace(-torch.pi, torch.pi, length + 1)
    z = torch.exp(1j * t)
    b = torch.ones(length + 1, dtype=torch.complex64)
    
    # Compute the Blaschke product for each pole
    for pole in poles:
        b = b * (z - pole) / (1 - torch.conj(pole) * z)
    
    return b

"""
Values of the argument function of a Blaschke product.

:param a: parameters of the Blaschke product, one-dimensional Tensor with complex numbers
:type a: Tensor
:param t: values in [-pi,pi), where the function values are needed, one-dimensional Tensor with floats
:type t: Tensor

:returns: the values of the argument function at the points in "t", one-dimensional Tensor with floats
:rtype: Tensor
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

"""
Inverse images by the argument function of a Blaschke product.

:param a: parameters of the Blaschke product, one-dimensional Tensor with complex numbers
:type a: Tensor
:param b: values in [-pi,pi), where the inverse images are needed, one-dimensional Tensor with floats
:type b: Tensor
:param epsi: required precision for the inverses (optional, default 1e-4)
:type epsi: float

:returns: the inverse images of the values in "b" by the argument function, one-dimensional Tensor with floats
:rtype: Tensor
"""
def arg_inv(a:torch.Tensor, b:torch.Tensor, epsi=1e-4) -> torch.Tensor:
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError('Parameters should be 1-D tensors!')
    if torch.max(torch.abs(a)) >= 1:
        raise ValueError('Bad poles!')

    if len(a) == 1:
        t = __arg_inv_one(a, b)
    else:
        t = __arg_inv_all(a, b, epsi)
    return t
"""
Inverse when the number of poles is 1
"""
def __arg_inv_one(a:torch.Tensor, b:torch.Tensor)->torch.Tensor:
    r = torch.abs(a)
    fi = torch.angle(a)
    mu = (1 + r) / (1 - r)

    gamma = 2 * torch.atan((1 / mu) * torch.tan(fi / 2))

    t = 2 * torch.atan((1 / mu) * torch.tan((b - gamma) / 2)) + fi
    t = (t + torch.pi) % (2 * torch.pi) - torch.pi  # move it in [-pi,pi)
    return t
"""
Inverse when the number of poles is greater than 1.
Uses the bisection method with an enhanced order of calculation
"""
def __arg_inv_all(a:torch.Tensor, b:torch.Tensor, epsi:float)->torch.Tensor:
    n = len(b)
    s = other.bisection_order(n) + 1
    x = torch.zeros(n+1)
    for i in range(1, n+2):
        if i == 1:
            v1 = -torch.pi
            v2 = torch.pi
            fv1 = -torch.pi  # fv1 <= y
            fv2 = torch.pi   # fv2 >= y
        elif i == 2:
            x[n] = x[0] + 2 * torch.pi  # x(s(2,1))
            continue
        else:
            v1 = x[s[i, 1]]
            v2 = x[s[i, 2]]
            fv1 = arg_fun(a, v1)
            fv2 = arg_fun(a, v2)

        ba = b[s[i, 0]]
        if fv1 == ba:
            x[s[i, 0]] = v1
            continue
        elif fv2 == ba:
            x[s[i, 0]] = v2
            continue
        else:
            xa = (v1 + v2) / 2
            fvk = arg_fun(a, torch.tensor(xa))  # Convert xa to a tensor
            while torch.abs(fvk - ba) > epsi:
                if fvk == ba:
                    x[s[i, 0]] = xa
                    break
                elif fvk < ba:
                    v1 = xa
                else:
                    v2 = xa
                xa = (v1 + v2) / 2
                fvk = arg_fun(a, torch.tensor(xa))  # Convert xa to a tensor
            x[s[i, 0]] = xa
    return x[:n]

"""
Inverse images by the argument function of a Blaschke product. Unlike arg_inv, this is "continuous on IR".

:param a: parameters of the Blaschke product, one-dimensional Tensor with complex numbers
:type a: Tensor
:param b: values in [-pi,pi), where the inverse images are needed, one-dimensional Tensor with floats
:type b: Tensor
:param epsi: required precision for the inverses (optional, default 1e-4)
:type epsi: float

:returns: the inverse images of the values in "b" by the argument function, one-dimensional Tensor with floats
:rtype: Tensor
"""
def argdr_inv(a:torch.Tensor, b:torch.Tensor, epsi=1e-4) -> torch.Tensor:
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError('Parameters should be 1-D tensors!')
    if torch.max(torch.abs(a)) >= 1:
        raise ValueError('Bad poles!')

    if len(a) == 1:
        t = __argdr_inv_one(a, b)
    else:
        t = __arg_inv_all(a, b, epsi)
    return t
"""
Inverse when the number of poles is 1
"""
def __argdr_inv_one(a:torch.Tensor, b:torch.Tensor)->torch.Tensor:
    r = torch.abs(a)
    fi = torch.angle(a)
    mu = (1 + r) / (1 - r)

    gamma = 2 * torch.atan((1 / mu) * torch.tan(fi / 2))

    t = 2 * torch.atan((1 / mu) * torch.tan((b - gamma) / 2)) + fi
    t = (t + torch.pi) % (2 * torch.pi) - torch.pi  # move it in [-pi,pi)
    return t
"""
Inverse when the number of poles is greater than 1.
Uses the bisection method with an enhanced order of calculation
"""
def __argdr_inv_all(a:torch.Tensor, b:torch.Tensor, epsi:float)->torch.Tensor:
    n = len(b)
    s = other.bisection_order(n) + 1
    x = torch.zeros(n+1)
    for i in range(1, n+2):
        if i == 1:
            v1 = -torch.pi
            v2 = torch.pi
            fv1 = -torch.pi  # fv1 <= y
            fv2 = torch.pi   # fv2 >= y
        elif i == 2:
            x[n] = x[0] + 2 * torch.pi  # x(s(2,1))
            continue
        else:
            v1 = x[s[i, 1]]
            v2 = x[s[i, 2]]
            fv1 = (argdr_fun(a, v1)-v1/2)/a.size(0)
            fv2 = (argdr_fun(a, v2)-v2/2)/a.size(0)

        ba = b[s[i, 0]]
        if fv1 == ba:
            x[s[i, 0]] = v1
            continue
        elif fv2 == ba:
            x[s[i, 0]] = v2
            continue
        else:
            xa = (v1 + v2) / 2
            fvk = (argdr_fun(a, torch.tensor(xa))-torch.tensor(xa/2))/a.size(0)
            while torch.abs(fvk - ba) > epsi:
                if fvk == ba:
                    x[s[i, 0]] = xa
                    break
                elif fvk < ba:
                    v1 = xa
                else:
                    v2 = xa
                xa = (v1 + v2) / 2
                fvk = (argdr_fun(a, torch.tensor(xa))-torch.tensor(xa/2))/a.size(0)
            x[s[i, 0]] = xa
    return x[:n]