import torch
from numpy import pi as PI
"""
Generates the Malmquist-Takenaka system.

:param length: number of points in case of uniform sampling
:type length: int
:param poles: poles of the system, one-dimensional Tensor with complex numbers
:type poles: Tensor

:returns: the elements of the MT system at the uniform sampling points as row vectors
:rtype: Tensor

"""
def mt_system(length:int, poles:torch.Tensor)->torch.Tensor:
    poles = poles.to(torch.complex64)
    if poles.ndim != 1 or length < 2:
        raise ValueError('Wrong parameters!')
    if torch.max(torch.abs(poles)) >= 1:
        raise ValueError('Bad poles!')

    mts = torch.zeros((poles.numel(), length), dtype=torch.complex64)
    t = torch.linspace(-PI, PI, length+1)[:-1]
    z = torch.exp(1j * t)

    fi = torch.ones(length, dtype=torch.complex64)  # the product defining MT elements so far
    for j in range(poles.numel()):
        co = torch.sqrt(1 - torch.abs(poles[j])**2)
        rec = 1 / (1 - torch.conj(poles[j]) * z)
        lin = co * rec
        bla = (z - poles[j]) * rec
        mts[j, :] = lin * fi
        fi = fi * bla

    return mts
