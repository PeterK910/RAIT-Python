import torch
import math
from collections.abc import Callable

from blaschke import arg_inv, argdr_inv
from mt_sys import mt_system
from biort_sys import biort_system
from rat_sys import lf_system, mlf_system

def addimag(v: torch.Tensor) -> torch.Tensor:
    """
    Calculates the imaginary part of v using FFT to be in Hardy space.

    Parameters
    ----------
    v : torch.Tensor
        A vector with real elements.

    Returns
    -------
    torch.Tensor
        A complex vector with appropriate imaginary part.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if v.ndim != 1:
        raise ValueError('v must be a 1-dimensional torch.Tensor.')
    
    if not torch.all(v.imag == 0):
        raise ValueError('The vector is not real!')

    # Calculate the imaginary part using FFT
    vf = torch.fft.fft(v)
    vif = __mt_arrange(vf)
    vi = torch.fft.ifft(vif)

    return vi

def __mt_arrange(t: torch.Tensor) -> torch.Tensor:
    """
    Rearrange FFT(v) so that lots of zeros appear on the right side of the FFT.

    Parameters
    ----------
    t : torch.Tensor
        The FFT of a vector.

    Returns
    -------
    torch.Tensor
        The rearranged FFT of a vector.
    """

    mt = t.size(0)
    ta = torch.zeros_like(t)
    ta[0] = t[0]
    
    for i in range(1, mt // 2):
        ta[i] = t[i] + torch.conj(t[mt - i])
        ta[mt - i] = 0

    return ta

"""
Gives a better order for multiple bisection runs

:param n: number of points
:type n: int

:returns: a 3-by-(n+1) tensor with the order of calculation, elements are integers
:rtype: Tensor
"""
def bisection_order(n: int) -> torch.Tensor:
    bo = torch.zeros((n+1, 3), dtype=torch.int32)
    bo[0, :] = torch.tensor([0, -1, -1])
    bo[1, :] = torch.tensor([n, -1, -1])
    bo[2, :] = torch.tensor([n // 2, 0, n])

    watch = 2 # which column is currently watched
    fill = 3 # where to fill the new values

    #fill the matrix with the ordering
    while fill <= n:
        #names
        ch = bo[watch,0] #child
        p1 = bo[watch,1] #parent 1
        p2 = bo[watch,2] #parent 2
        #INVAR: p1 < ch < p2

        #if there is place for another element...
        #the child with parent 1
        if ch - p1 > 1 and fill <= n:
            gch = math.floor((ch + p1)/2) #grandchild
            bo[fill] = torch.tensor([gch, p1, ch])
            fill += 1
        #the child with parent 2
        if p2 - ch > 1 and fill <= n:
            gch = math.floor((ch + p2)/2) #grandchild
            bo[fill] = torch.tensor([gch, ch, p2])
            fill += 1
        
        #watch the next column
        watch += 1
    return bo

"""
Computes the non-equidistant complex discretization on the unit disc that refers to the given poles.

:param mpoles: poles of the rational system
:type mpoles: Tensor
:param eps: Accuracy of the complex discretization on the unit disc, by default 1e-6.
:type eps: float

:returns: arguments of the poles
:rtype: Tensor
"""
def discretize_dc(mpoles: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    if torch.max(torch.abs(mpoles)) >= 1:
        raise ValueError("Poles must be inside the unit disc")

    m = mpoles.numel()
    z = torch.linspace(-torch.pi, torch.pi, m+1)
    t = arg_inv(mpoles, z, eps)
    return t


def discretize_dr(mpoles: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """
    Computes the non-equidistant real discretization on the unit disc that refers to the given poles.

    Parameters
    ----------
    mpoles : torch.Tensor
        Poles of the Blaschke product, expected to be a 1D tensor.
    eps : float, optional
        Accuracy of the real discretization on the unit disc, by default 1e-6.

    Returns
    -------
    torch.Tensor
        Non-equidistant real discretization as a 1D tensor.

    Raises
    ------
    ValueError
        If the poles are not inside the unit disc.
    """
    if torch.max(torch.abs(mpoles)) >= 1:
        raise ValueError("Poles must be inside the unit disc")

    mpoles = torch.cat((torch.tensor([0.0]), mpoles))
    m = mpoles.size(0)
    z = torch.linspace(-(m-1)*torch.pi, (m-1)*torch.pi, steps=m)
    z = z / m
    t = argdr_inv(mpoles, z, eps)
    return t

def dotdc(F: torch.Tensor, G: torch.Tensor, poles: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Computes complex discrete dot product of two functions in H^2(ID).

    Parameters
    ----------
    F : torch.Tensor
        Values of the first function (ID -> IC) on the unit disk.
    G : torch.Tensor
        Values of the second function (ID -> IC) on the unit disk.
    poles : torch.Tensor
        Poles of the rational system.
    t : torch.Tensor
        Arguments for which to evaluate the dot product.

    Returns
    -------
    torch.Tensor
        Values of the complex dot product of 'F' and 'G' at 't'.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(F, torch.Tensor) or not isinstance(G, torch.Tensor):
        raise ValueError('F and G must be torch.Tensors.')
    
    if not isinstance(poles, torch.Tensor) or poles.ndim != 1:
        raise ValueError('Poles must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(t, torch.Tensor) or t.ndim != 1:
        raise ValueError('t must be a 1-dimensional torch.Tensor.')

    if F.size(0) != G.size(0) or F.size(0) != t.size(0):
        raise ValueError('F, G, and t must have the same number of elements.')

    # Compute the kernel values
    kernel_vals = kernel(torch.exp(1j * t[:-1]), torch.exp(1j * t[:-1]), poles)

    # Compute the complex discrete dot product
    s = torch.sum(F[:-1] * torch.conj(G[:-1]) / kernel_vals)

    return s


import torch

def dotdr(F: torch.Tensor, G: torch.Tensor, mpoles: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Computes discrete real dot product of two functions in H^2(ID).

    Parameters
    ----------
    F : torch.Tensor
        Values of the first function (ID -> IC) on the unit disk.
    G : torch.Tensor
        Values of the second function (ID -> IC) on the unit disk.
    mpoles : torch.Tensor
        Poles of the rational system excluding zero.
    t : torch.Tensor
        Arguments for which to evaluate the dot product.

    Returns
    -------
    torch.Tensor
        Values of the real dot product of 'F' and 'G' at 't'.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(F, torch.Tensor):
        raise ValueError('F must be a torch.Tensor.')
    
    if not isinstance(G, torch.Tensor):
        raise ValueError('G must be a torch.Tensor.')
    
    if not isinstance(mpoles, torch.Tensor) or mpoles.ndim != 1:
        raise ValueError('mpoles must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(t, torch.Tensor) or t.ndim != 1:
        raise ValueError('t must be a 1-dimensional torch.Tensor.')

    if F.size(0) != G.size(0):
        raise ValueError('F and G must have the same number of elements.')

    # Prepend zero to the poles
    mpoles_with_zero = torch.cat((torch.zeros(1), mpoles))

    # Compute the kernel values
    kernel_vals = kernel(torch.exp(1j * t), torch.exp(1j * t), mpoles_with_zero)

    # Compute the real discrete dot product
    s = torch.sum(F * torch.conj(G) / (2 * torch.real(kernel_vals) - 1))

    return s





def kernel(y:torch.Tensor,z:torch.Tensor,mpoles: torch.Tensor) -> torch.Tensor:
    """
    Computes the weight function of discrete dot product in H^2(D).

    Parameters
    ----------
    y : torch.Tensor
        First argument.
    z : torch.Tensor
        Second argument.
    mpoles : torch.Tensor
        Poles of the rational system.

    Returns
    -------
    torch.Tensor
        Value of the weight function at arguments "y" and "z".
    """
    r = torch.zeros_like(y)
    m = len(mpoles)
    if y == z:
        for k in range(m):
            alpha = torch.angle(mpoles[k])
            R = torch.abs(mpoles[k])
            t = torch.angle(z)
            r += __poisson(R, t - alpha)
    else:
        for i in range(1, m + 1):
            r += __MT(i - 1, mpoles, y) * torch.conj(__MT(i - 1, mpoles, z))
    return r

def __poisson(r:torch.Tensor,t:torch.Tensor) -> torch.Tensor:
    """
    Compute the values of the poisson function at (r,t).

    Parameters
    ----------
    r : torch.Tensor
        The radial distance.
    t : torch.Tensor
        The angle in radians.

    Returns
    -------
    torch.Tensor
        The calculated Poisson ratio.
    """
    return (1-r**2)/(1-2*r*math.cos(t)+r**2)

def __MT(n: int, mpoles: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
     Compute the values of the n-th Malmquist-Takenaka function at z.

    Parameters:
    ----------
        n : int
            The order of the Malmquist-Takenaka function.
        mpoles : torch.Tensor
            Poles of the rational system.
        z : torch.Tensor
            The input tensor.

    Returns:
    -------
        torch.Tensor
            Values of the Malmquist-Takenaka function.
    """
    r = torch.ones_like(z)
    for k in range(n):
        r *= (z - mpoles[k]) / (1 - torch.conj(mpoles[k]) * z)
    r *= math.sqrt(1 - torch.abs(mpoles[n]) ** 2 / (1 - torch.conj(mpoles[n]) * z))
    return r

"""
Returns the multiplicity of all elements of the tensor 'mpoles'.

:param mpoles: poles with arbitrary multiplicities
:type mpoles: Tensor

:returns: unique elements of 'mpoles' and their multiplicities
:rtype: tuple[Tensor, Tensor]
"""
def multiplicity(mpoles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unique, counts = torch.unique(torch.tensor(mpoles), return_counts=True)
    return unique, counts

def subsample(sample:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
    """
    TODO: check if interpolation without numpy is correct here
    Interpolate values between uniform sampling points using linear interpolation.

    Parameters
    ----------
    sample : torch.Tensor
        A 1D tensor of uniformly sampled values on [-pi, pi).
    x : torch.Tensor
        The values at which interpolation is to be computed.

    Returns
    -------
    torch.Tensor
        The interpolated values at the points specified by x.
    """

    # Validate input types
    if not isinstance(sample, torch.Tensor):
        raise TypeError("sample must be a torch.Tensor")
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")

    # Validate input dimensions
    if sample.dim() != 1:
        raise ValueError("sample must be a 1D tensor")
    if x.dim() != 1:
        raise ValueError("x must be a 1D tensor")

    # Number of samples
    len = sample.size(0)

    # Create a tensor of sample points
    sx = torch.linspace(-torch.pi, torch.pi, len)

    # Reshape x to (N, 1, 1) and sx to (1, len, 1) for broadcasting
    x_reshaped = x.view(-1, 1, 1)
    sx_reshaped = sx.view(1, -1, 1)

    # Compute the ratio for interpolation
    ratio = (x_reshaped - sx_reshaped) / (2 * torch.pi / len)

    # Use torch.nn.functional.interpolate for interpolation
    y = torch.nn.functional.interpolate(sample.view(1, 1, -1), scale_factor=ratio, mode='linear', align_corners=True)

    # Squeeze to remove extra dimensions
    return y.squeeze()

def coeff_conv(length:int, poles:torch.Tensor, coeffs:torch.Tensor, base1:str, base2:str) -> torch.Tensor:
    """
    Convert the coefficients between the continuous systems base1 and base2.

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling.
    poles : torch.Tensor
        Poles of the continuous systems.
    coeffs : torch.Tensor
        Coefficients with respect to the continuous system 'base1'.
    base1 : str
        Type of the continuous system to be converted.
    base2 : str
        Type of the converted continuous system.

    Returns
    -------
    torch.Tensor
        Converted coefficients with respect to the system 'base2'.
    
    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    
    # Validate input parameters
    if poles.size(0) != 1:
        raise ValueError("poles must be a 1D tensor")
    if length < 2:
        raise ValueError("length must be an integer greater than or equal to 2.")
    if torch.max(torch.abs(poles)) >= 1:
        raise ValueError('Poles must be inside the unit circle!')
    
    if coeffs.size(0) != 1:
        raise ValueError('Coeffs should be row vector!')
    
    # Helper function
    def get_system(base, length, poles):
        if base == 'lf':
            return lf_system(length, poles)
        elif base == 'mlf':
            return mlf_system(length, poles)
        elif base == 'biort':
            return biort_system(length, poles)
        elif base == 'mt':
            return mt_system(length, poles)
        else:
            raise ValueError('Invalid system type! Choose from lf, mlf, biort, mt.')

    # Get systems for base1 and base2
    g1 = get_system(base1, length, poles)
    g2 = get_system(base2, length, poles)
    #TODO: check if the lines below are correct
    # Perform matrix operations
    F = g1.mm(g1.t()) / length
    G = g1.mm(g2.t()) / length
    
    # Solve linear system and return converted coefficients
    co = torch.linalg.solve(G, F.mm(coeffs.t())).t()
    
    return co

import torch

def coeffd_conv(poles: torch.Tensor, coeffs: torch.Tensor, base1: str, base2: str, eps: float = 1e-6) -> torch.Tensor:
    """
    Converts the coefficients between the discrete systems base1 and base2.

    Parameters
    ----------
    poles : torch.Tensor
        Poles of the discrete systems (1-dimensional tensor).
    coeffs : torch.Tensor
        Coefficients with respect to the discrete system 'base1' (1-dimensional tensor).
    base1 : str
        Type of the discrete system to be converted.
    base2 : str
        Type of the converted discrete system.
    eps : float, optional
        Accuracy of the discretization on the unit disc (default is 1e-6).

    Returns
    -------
    torch.Tensor
        Converted coefficients with respect to the system 'base2'.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(poles, torch.Tensor) or poles.ndim != 1:
        raise ValueError('Poles must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(coeffs, torch.Tensor) or coeffs.ndim != 1:
        raise ValueError('Coeffs should be a 1-dimensional torch.Tensor.')
    
    if not isinstance(base1, str) or not isinstance(base2, str):
        raise ValueError('Base1 and Base2 must be strings.')
    
    if not isinstance(eps, float):
        raise ValueError('Eps must be a float.')
    
    if torch.max(torch.abs(poles)) >= 1:
        raise ValueError('Poles must be inside the unit circle!')

    if base1 not in ['mlfdc', 'biortdc', 'mtdc']:
        raise ValueError('Invalid system type for base1! Choose from mlfdc, biortdc, mtdc.')
    
    if base2 not in ['mlfdc', 'biortdc', 'mtdc']:
        raise ValueError('Invalid system type for base2! Choose from mlfdc, biortdc, mtdc.')
    
    # Generate systems based on 'base1' and 'base2'
    g1 = globals()[f'{base1}_system'](poles, eps)
    g2 = globals()[f'{base2}_system'](poles, eps)

    # Convert coefficients between systems
    F = g1 @ g1.t() / coeffs.shape[0]
    G = g1 @ g2.t() / coeffs.shape[0]
    
    co = torch.linalg.solve(G.t(), F @ coeffs.unsqueeze(0).t()).squeeze()

    return co

