import torch
import math
from scipy.signal.windows import tukey
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt

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


def bisection_order(n: int) -> torch.Tensor:
    """
    Gives a better order for multiple bisection runs.

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    torch.Tensor
        A 3-by-(n+1) matrix with the order of calculation.

    Raises
    ------
    ValueError
        If input parameter is invalid.
    """
    # Validate input parameter
    if not isinstance(n, int) or n < 0:
        raise ValueError('n must be a non-negative integer.')

    # Initialize the matrix
    bo = torch.zeros(n + 1, 3)
    bo[0, :] = torch.tensor([0, -1, -1])
    bo[1, :] = torch.tensor([n, -1, -1])
    bo[2, :] = torch.tensor([n // 2, 0, n])

    watch = 3  # Column currently being watched
    fill = 4  # Where to fill the new values

    # Fill the matrix with the ordering
    while fill <= n + 1:
        # Get child and parents
        ch = bo[watch, 0]
        p1 = bo[watch, 1]
        p2 = bo[watch, 2]

        # Add child with parent 1
        if ch - p1 > 1 and fill <= n + 1:
            gch = (ch + p1) // 2  # Grandchild
            bo[fill, :] = torch.tensor([gch, p1, ch])
            fill += 1

        # Add child with parent 2
        if p2 - ch > 1 and fill <= n + 1:
            gch = (ch + p2) // 2  # Grandchild
            bo[fill, :] = torch.tensor([gch, ch, p2])
            fill += 1

        watch += 1

    return bo




def discretize_dc(mpoles: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the non-equidistant complex discretization on the unit disc
    that refers to the given poles.

    Parameters
    ----------
    mpoles : torch.Tensor
        Poles of the Blaschke product.
    eps : float, optional
        Accuracy of the complex discretization on the unit disc (default: 1e-6).

    Returns
    -------
    torch.Tensor
        Non-equidistant complex discretization.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    # Validate input parameters
    if not isinstance(mpoles, torch.Tensor) or mpoles.ndim != 1:
        raise ValueError('mpoles must be a 1-dimensional torch.Tensor.')

    if torch.max(torch.abs(mpoles)) >= 1:
        raise ValueError('Poles must be inside the unit circle!')

    # Calculate the non-equidistant complex discretization
    m = len(mpoles)
    z = torch.linspace(-np.pi, np.pi, m + 1)
    t = arg_inv(mpoles, z, eps)  # Assuming arg_inv is a helper function

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

def multiplicity(mpoles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the multiplicity of all elements of the vector 'mpoles'.

    Parameters
    ----------
    mpoles : torch.Tensor
        Poles with arbitrary multiplicities.

    Returns
    -------
    torch.Tensor
        Vector of the poles that contains a pole only once.
    torch.Tensor
        Mult(i) refers to the multiplicity of the ith pole.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    if not isinstance(mpoles, torch.Tensor) or mpoles.ndim != 1:
        raise ValueError('mpoles must be a 1-dimensional torch.Tensor.')
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

def periodize(v: torch.Tensor, alpha: float, draw: bool = False) -> torch.Tensor:
    """
    Calculates the periodized extension of a signal.

    Parameters
    ----------
    v : torch.Tensor
        An arbitrary vector.
    alpha : float
        Alpha parameter of the Tukey window.
    draw : bool, optional
        Logical value to display the periodized signal (default is False).

    Returns
    -------
    torch.Tensor
        Periodized signal.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(v, torch.Tensor) or v.ndim != 1:
        raise ValueError('Signal must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(alpha, float) or not (0 <= alpha <= 1):
        raise ValueError('Alpha must be a float between 0 and 1.')
    
    # Calculate the periodized signal
    v_max = torch.max(v)
    v_ind = torch.argmax(v)
    gap = (torch.max(v[0], v[-1]) + torch.min(v[0], v[-1])) / 2
    v = v - gap
    n = v.size(0)
    N = int(math.ceil(n * alpha))

    # Approximating derivatives
    smooth = savgol_filter(v, 11, 2)
    smooth = torch.from_numpy(smooth)
    fx = __remNEP(smooth)
    _, dy, _ = __curvatures(torch.arange(1, n + 1), smooth)  # This would be a custom function or imported from elsewhere

    left = torch.sum(dy[0:fx[1]])
    right = torch.sum(dy[fx[-2]:])

    s = torch.cat((torch.ones(N) * v[0], v, torch.ones(N) * v[-1]))

    # Further processing based on derivatives and signal values

    if torch.sign(left) == torch.sign(right):
        end_slope = (v[-1] - v[0]) / n
        if torch.sign(left) == torch.sign(end_slope):
            # taking into account the significance of the first derivatives
            # at the end points
            trs = abs(torch.max(v[0], v[-1]) - torch.min(v[0], v[-1])) / 2
            if abs(left) < abs(right):
                s -= torch.sign(right) * (abs(v[-1]) + trs)
            elif abs(left) > abs(right):
                s += torch.sign(left) * (abs(v[0]) + trs)
        else:
            avg = (v[0] + v[-1]) / 2           
            s -= avg
    
    elif torch.sign(left) > 0:
        # positive first derivatives
        if v[0] < 0 or v[-1] < 0:
            s -= torch.min(v[0], v[-1])
    
    else:
        # negative first derivatives
        if torch.sign(v[0]) > 0 or torch.sign(v[-1]) > 0:
            s -= torch.max(v[0], v[-1])

    # Apply Tukey window
    tukey_window = tukey(len(s), (2 * N + 2) / len(s))
    s = s * torch.from_numpy(tukey_window)

    s = s + (v_max - s[N + v_ind])

    if draw:
        plt.plot(s.numpy(), 'r')
        plt.plot(range(N, N + len(v)), s.numpy()[N:N + len(v)], 'g')
        plt.show()

    return s

def __remNEP(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Remove Non Extreme Points from the given data array.

    Parameters
    ----------
    data : torch.Tensor
        A 1-dimensional tensor of data points.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Indices of extreme points and their corresponding values.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(data, torch.Tensor) or data.ndim != 1:
        raise ValueError('Data must be a 1-dimensional torch.Tensor.')

    n = data.size(0)
    j = 0
    x = torch.zeros(n, dtype=torch.int64)
    y = torch.zeros(n, dtype=data.dtype)
    x[j] = 0
    y[j] = data[j]
    
    for i in range(1, n - 1):
        # Check if data[i] is a min or max extreme point
        if (y[j] < data[i] and data[i] > data[i + 1]) or (y[j] > data[i] and data[i] < data[i + 1]):
            j += 1
            y[j] = data[i]
            x[j] = i
    
    j += 1
    y[j] = data[n - 1]
    x[j] = n - 1
    
    # Trim the tensors to the size of j + 1
    x = x[:j + 1]
    y = y[:j + 1]

    return x, y

def __curvatures(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Approximate discrete curvatures.

    Parameters
    ----------
    x : torch.Tensor
        A 1-dimensional tensor of x-coordinates.
    y : torch.Tensor
        A 1-dimensional tensor of y-coordinates corresponding to x.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Curvatures, first derivatives, and second derivatives.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(x, torch.Tensor) or x.ndim != 1:
        raise ValueError('x must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(y, torch.Tensor) or y.ndim != 1:
        raise ValueError('y must be a 1-dimensional torch.Tensor.')
    
    if x.size(0) != y.size(0):
        raise ValueError('x and y must have the same number of elements.')

    n = y.size(0)
    dy = torch.zeros(n - 2, dtype=y.dtype)
    ddy = torch.zeros(n - 2, dtype=y.dtype)
    
    # Curvatures are set to zero at the endpoints by default.
    k = torch.zeros(n - 2, dtype=y.dtype)
    
    # Bezier-approximation
    for i in range(1, n - 1):
        t = x[i]
        ddy[i - 1] = (y[i - 1] * (-2 / (x[i + 1] - x[i - 1])) * (-1 / (x[i + 1] - x[i - 1])) 
                      - 2 * (4 * y[i] - y[i - 1] - y[i + 1]) / (x[i + 1] - x[i - 1])**2 
                      + y[i + 1] * (2 / (x[i + 1] - x[i - 1])**2))
        
        dy[i - 1] = (2 * y[i - 1] * ((x[i + 1] - t) / (x[i + 1] - x[i - 1])) * (-1 / (x[i + 1] - x[i - 1])) 
                     + (4 * y[i] - y[i - 1] - y[i + 1]) * (x[i - 1] + x[i + 1] - 2 * t) / (x[i + 1] - x[i - 1])**2 
                     + 2 * y[i + 1] * ((t - x[i - 1]) / (x[i + 1] - x[i - 1])) * (1 / (x[i + 1] - x[i - 1])))
        
        k[i - 1] = ddy[i - 1] / ((1 + dy[i - 1]**2)**(3/2))

    return k, dy, ddy
