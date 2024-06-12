import torch
from other import discretize_dc
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
        raise ValueError('Poles must be inside the unit disc!')

    mts = torch.zeros((poles.numel(), length), dtype=torch.complex64)
    t = torch.linspace(-torch.pi, torch.pi, length+1)[:-1]
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

def mtdc_system(mpoles:torch.Tensor, eps:float=1e-6) -> torch.Tensor:
    """
    Generates the discrete complex MT system.

    Parameters
    ----------
    mpoles : torch.Tensor
        Poles of the discrete complex MT system.
    eps : float, optional
        Accuracy of the discretization on the unit disc (default is 1e-6).

    Returns
    -------
    torch.Tensor
        The elements of the discrete complex MT system at the uniform sampling points as row vectors.
    
    Raises
    ------
    ValueError
        If the poles are not inside the unit disc.
    """
    if not isinstance(mpoles, torch.Tensor):
        raise TypeError("mpoles must be a torch.Tensor")
    if not isinstance(eps, float):
        raise TypeError("eps must be a float")

    if torch.max(torch.abs(mpoles)) >= 1:
        raise ValueError("Poles must be inside the unit disc")

    m = mpoles.numel()
    t = discretize_dc(mpoles, eps)
    mts = torch.zeros(m, m+1, dtype=mpoles.dtype)

    for j in range(m):
        mts[j, :] = __mt(j, mpoles, torch.exp(1j * t))

    return mts

def __mt(n:int, mpoles:torch.Tensor, z:torch.Tensor) -> torch.Tensor:
    """
    Compute the values of the nth Malmquist-Takenaka function at z.

    Parameters
    ----------
    n : int
        The order of the Malmquist-Takenaka function.
    mpoles : torch.Tensor
        Poles of the discrete complex MT system.
    z : torch.Tensor
        The complex points where the MT function is evaluated.

    Returns
    -------
    torch.Tensor
        The calculated values of the nth Malmquist-Takenaka function.
    """
    if not isinstance(n, int):
        raise TypeError("n must be an int")
    if not isinstance(mpoles, torch.Tensor):
        raise TypeError("mpoles must be a torch.Tensor")
    if not isinstance(z, torch.Tensor):
        raise TypeError("z must be a torch.Tensor")

    r = torch.ones_like(z)
    for k in range(1, n+1):
        r *= (z - mpoles[k-1]) / (1 - mpoles[k-1].conj() * z)
    r *= torch.sqrt(1 - torch.abs(mpoles[n])**2) / (1 - mpoles[n].conj() * z)
    return r
