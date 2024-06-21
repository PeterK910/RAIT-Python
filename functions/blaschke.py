import torch
import matplotlib.pyplot as plt
from PIL import Image

from util import bisection_order
import torch

def arg_der(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Derivatives of the argument function of a Blaschke product.

    Parameters
    ----------
    a : torch.Tensor
        Parameters of the Blaschke product.
    t : torch.Tensor
        Values in [-pi, pi), where the function values are needed.

    Returns
    -------
    torch.Tensor
        The derivatives of the argument function at the points in t.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(a, torch.Tensor) or a.ndim != 1:
        raise ValueError('a must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(t, torch.Tensor) or t.ndim != 1:
        raise ValueError('t must be a 1-dimensional torch.Tensor.')
    
    if torch.max(torch.abs(a)) >= 1:
        raise ValueError('Elements of a must be inside the unit circle!')

    # Calculate derivatives
    bd = torch.zeros_like(t)
    
    for i in range(a.size(0)):
        bd += __arg_der_one(a[i], t)
    
    bd /= a.size(0)

    return bd

def __arg_der_one(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the derivative of the argument function for one element of a Blaschke product.

    Parameters
    ----------
    a : torch.Tensor
        A single parameter of the Blaschke product, must be a complex number inside the unit circle.
    t : torch.Tensor
        Values in [-pi, pi), where the function values are needed.

    Returns
    -------
    torch.Tensor
        The derivative of the argument function at the points in t for one element.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(a, torch.Tensor) or a.ndim != 0 or not torch.is_complex(a):
        raise ValueError('a must be a 0-dimensional complex torch.Tensor.')
    
    if not isinstance(t, torch.Tensor) or t.ndim != 1:
        raise ValueError('t must be a 1-dimensional torch.Tensor.')
    
    if a.abs() >= 1:
        raise ValueError('The absolute value of a must be inside the unit circle!')

    # Calculate the derivative for one element
    r = a.abs()
    fi = a.angle()

    bd = (1 - r**2) / (1 + r**2 - 2 * r * torch.cos(t - fi))

    return bd

def blaschkes(len: int, poles: torch.Tensor) -> torch.Tensor:
    """
    Gives a sampled Blaschke function.

    Parameters
    ----------
    len : int
        Number of points for uniform sampling.
    poles : torch.Tensor
        Parameters of the Blaschke product.

    Returns
    -------
    torch.Tensor
        The Blaschke product with given parameters sampled at uniform
        points on the torus.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    # Validate input parameters
    if not isinstance(len, int) or len <= 0:
        raise ValueError('len must be a positive integer.')

    if not isinstance(poles, torch.Tensor) or poles.ndim != 1:
        raise ValueError('Poles must be a 1-dimensional torch.Tensor.')

    if torch.max(torch.abs(poles)) >= 1:
        raise ValueError('Poles must be inside the unit circle!')

    # Calculate the sampled Blaschke function
    t = torch.linspace(-torch.pi, torch.pi, len + 1)
    z = torch.exp(1j * t)
    b = torch.ones(len + 1, dtype=torch.complex64)
    for p in poles:
        b = b * (z - p) / (1 - torch.conj(p) * z)

    return b

import torch
import numpy as np


def blaschkes_img(path: str, a: complex, show: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transforms an image by applying the Blaschke function defined by poles 'a'.

    Parameters
    ----------
    path : str
        Path of the input image.
    a : complex
        Parameters of the Blaschke function.
    show : bool
        Displays the transformed images.

    Returns
    -------
    torch.Tensor
        Absolute values of the Blaschke function.
    torch.Tensor
        Arguments of the Blaschke function.
    torch.Tensor
        Transformed image.
    
    Raises
    ------
    FileNotFoundError
        If the input image file does not exist.
    """
    try:
        img = Image.open(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file '{path}' not found.")

    # Convert image to grayscale
    M = np.array(img.convert("L"))

    B = torch.full_like(torch.tensor(M), 0.5)
    B_arg = torch.full_like(torch.tensor(M), 0.5)
    B_abs = torch.full_like(torch.tensor(M), 0.5)

    m, n = M.shape
    half = m // 2
    bound = half - 10

    # Blaschke function
    def blaschke(z: complex) -> complex:
        return (z - a) / (1 - np.conj(a) * z)
    
    for i in range(m):
        for j in range(n):
            if np.sqrt((j - half) ** 2 + (i - half) ** 2) < bound:
                z = ((j - half) / bound) + ((i - half) * 1j / bound)
                b = blaschke(z)
                I = round(bound * np.real(b) + half)
                J = round(bound * np.imag(b) + half)
                B[i, j] = M[J, I]
                B_abs[i, j] = abs(b)
                t = np.arctan2(np.imag(b), np.real(b))
                B_arg[i, j] = t
            else:
                B[i, j] = np.nan

    if show:
        # Display transformed images
        plt.figure()
        plt.imshow(B, cmap="gray")
        plt.title("Transformed image")

        plt.figure()
        plt.imshow(B_abs, cmap="jet")
        plt.title("Absolute value of Blaschke function")

        plt.figure()
        plt.imshow(B_arg, cmap="jet")
        plt.title("Argument function")
        plt.show()

    return B_abs, B_arg, B

def arg_fun(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the values of the argument function of a Blaschke product.

    Parameters
    ----------
    a : torch.Tensor
        Parameters of the Blaschke product.
    t : torch.Tensor
        Values in [-pi, pi), where the function values are needed.

    Returns
    -------
    torch.Tensor
        The values of the argument function at the points in t.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    # Validate input parameters
    if not isinstance(a, torch.Tensor) or a.ndim != 1:
        raise ValueError('a must be a 1-dimensional torch.Tensor.')

    if not isinstance(t, torch.Tensor) or t.ndim != 1:
        raise ValueError('t must be a 1-dimensional torch.Tensor.')

    if torch.max(torch.abs(a)) >= 1:
        raise ValueError('Elements of a must be inside the unit circle!')

    # Calculate the argument function values
    b = torch.zeros(len(t))
    for i in range(len(a)):
        b += __arg_fun_one(a[i], t)
    b /= len(a)

    return b

def __arg_fun_one(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r = abs(a)
    fi = torch.angle(a)
    mu = (1 + r) / (1 - r)

    gamma = 2 * torch.atan((1 / mu) * torch.tan(fi / 2))

    b = 2 * torch.atan(mu * torch.tan((t - fi) / 2)) + gamma
    b = torch.fmod(b + torch.pi, 2 * torch.pi) - torch.pi  # move it in [-pi, pi)
    return b

def argdr_fun(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    #TODO: are function parameters really the same as in arg_fun?
    """
    Calculate the continuous argument function on the interval IR for given 'a' and 't'.

    Parameters
    ----------
    a : torch.Tensor
        Parameters of the Blaschke product.
    t : torch.Tensor
        Values in [-pi, pi), where the function values are needed.

    Returns
    -------
    torch.Tensor
        The result of the continuous argument function.
    """
    # Validate input parameters
    if not isinstance(a, torch.Tensor) or a.ndim != 1:
        raise ValueError('a must be a 1-dimensional torch.Tensor.')
    if not isinstance(t, torch.Tensor) or t.ndim != 1:
        raise ValueError('t must be a 1-dimensional torch.Tensor.')
    
    # Initialize the result tensor
    b = torch.zeros_like(t)
    
    # Calculate the continuous argument function
    for j in range(t.size(0)):
        for i in range(a.size(0)):
            bs = arg_fun(a[i], t[j])
            b[j] += bs + 2 * torch.pi * torch.floor((t[j] + torch.pi) / (2 * torch.pi))
    
    return b


def arg_inv(a: torch.Tensor, b: torch.Tensor, epsi: float = 1e-4) -> torch.Tensor:
    """
    Inverse images by the argument function of a Blaschke product.

    Parameters
    ----------
    a : torch.Tensor
        Parameters of the Blaschke product.
    b : torch.Tensor
        Values in [-pi, pi) whose inverse image is needed.
    epsi : float, optional
        Required precision for the inverses (default is 1e-4).

    Returns
    -------
    torch.Tensor
        Inverse images by the argument function of the points in 'b'.
    """
    # Validate input parameters
    if not isinstance(a, torch.Tensor) or a.ndim != 1:
        raise ValueError('a must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(b, torch.Tensor) or b.ndim != 1:
        raise ValueError('b must be a 1-dimensional torch.Tensor.')
    
    if torch.max(torch.abs(a)) >= 1:
        raise ValueError('Elements of a must be inside the unit circle!')
    
    if epsi <= 0:
        raise ValueError('epsi must be a positive number!')
    
    # Initialize the result tensor
    t = torch.zeros_like(b)
    
    # Calculate inverse images
    if len(a) == 1:
        t = __arg_inv_one(a, b)
    else:
        t = __arg_inv_all(a, b, epsi)
    
    return t

def __arg_inv_one(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Calculate the inverse images by the argument function of a Blaschke product.

    Parameters
    ----------
    a : torch.Tensor
        Parameters of the Blaschke product.
    b : torch.Tensor
        Values in [-pi, pi) whose inverse image is needed.

    Returns
    -------
    torch.Tensor
        Inverse images by the argument function of the points in 'b'.
    """
    # Validate input parameters
    if not isinstance(a, torch.Tensor) or a.ndim != 1:
        raise ValueError('a must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(b, torch.Tensor) or b.ndim != 1:
        raise ValueError('b must be a 1-dimensional torch.Tensor.')
    
    # Calculate intermediate variables
    r = torch.abs(a)
    fi = torch.angle(a)
    mu = (1 + r) / (1 - r)
    gamma = 2 * torch.atan((1 / mu) * torch.tan(fi / 2))
    
    # Calculate inverse images
    t = 2 * torch.atan((1 / mu) * torch.tan((b - gamma) / 2)) + fi
    t = torch.fmod(t + torch.pi, 2 * torch.pi) - torch.pi  # Move it in [-pi, pi)
    
    return t

def __arg_inv_all(a: torch.Tensor, b: torch.Tensor, epsi: float) -> torch.Tensor:
    """
    Calculate the inverse images by the argument function of a Blaschke product.

    Parameters
    ----------
    a : torch.Tensor
        Parameters of the Blaschke product.
    b : torch.Tensor
        Values in [-pi, pi) whose inverse image is needed.
    epsi : float
        Tolerance for the bisection method.

    Returns
    -------
    torch.Tensor
        Inverse images by the argument function of the points in 'b'.
    """
    # Validate input parameters
    if not isinstance(a, torch.Tensor) or a.ndim != 1:
        raise ValueError('a must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(b, torch.Tensor) or b.ndim != 1:
        raise ValueError('b must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(epsi, float) or epsi <= 0:
        raise ValueError('epsi must be a positive float.')
    
    # Initialize variables
    n = len(b)
    s = bisection_order(n) + 1
    x = torch.zeros(n)
    
    for i in range(n):
        if i == 0:
            v1, v2 = -torch.pi, torch.pi
            fv1, fv2 = -torch.pi, torch.pi

            v1 = torch.tensor(v1)
            v2 = torch.tensor(v2)
            fv1 = torch.tensor(fv1)
            fv2 = torch.tensor(fv2)
        elif i == 1:
            x[n] = x[0] + 2 * torch.pi
            continue
        else:
            v1, v2 = x[s[i, 1]], x[s[i, 2]]
            fv1, fv2 = arg_fun(a, v1), arg_fun(a, v2)
        
        ba = b[s[i, 0]]
        if fv1 == ba:
            x[s[i, 0]] = v1
            continue
        elif fv2 == ba:
            x[s[i, 0]] = v2
            continue
        else:
            xa = (v1 + v2) / 2
            fvk = arg_fun(a, xa)
            while abs(fvk - ba) > epsi:
                if fvk == ba:
                    x[s[i, 0]] = xa
                    return x
                elif fvk < ba:
                    v1 = xa
                else:
                    v2 = xa
                xa = (v1 + v2) / 2
                fvk = arg_fun(a, xa)
            x[s[i, 0]] = xa
    
    return x

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
        t = __argdr_inv_all(a, b, epsi)
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
    s = bisection_order(n) + 1
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
"""
Shows an animation related to the inverse of an equidistant discretization by an argument function

:param a: parameter of the Blaschke product, complex number
:type a: Tensor
:param n: number of points in the discretization
:type n: int

:returns: None
"""
def arg_inv_anim(a:torch.Tensor, n:int):
    if not (a.numel() == 1 and a.dtype == torch.complex64 and type(n) == int):
        raise ValueError('Please provide two numbers, first should be complex, second should be integer!')
    if a.abs() >= 1:
        raise ValueError('The parameter should be inside the unit disc!')

    t = torch.linspace(-torch.pi, torch.pi, n+1)[:-1]  # discretization

    anim = 32
    part = 2 * torch.pi / n / anim
    curr = 0

    plt.ion()  # Turn on interactive mode for animation
    fig, ax = plt.subplots()
    for i in range(anim * n): #number of frames
        b = arg_inv(a, t + curr)
        ax.plot(torch.cos(b).numpy(), torch.sin(b).numpy(), 'ko')
        ax.plot(a.real, a.imag, 'ro')
        ax.set_aspect('equal')
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        plt.draw()
        plt.pause(0.01)
        ax.cla()  # Clear the plot for the next frame

        curr += part
        if curr > 2 * torch.pi / n:
            curr -= 2 * torch.pi / n

    plt.ioff()  # Turn off interactive mode