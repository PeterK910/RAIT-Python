import torch
import math
from scipy.signal.windows import tukey
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from torchinterp1d import interp1d

def conj_trans(v: torch.Tensor) -> torch.Tensor:
    """
    Transpose and conjugate the input tensor.
    
    TODO: test this function, even though it is simple

    Parameters
    ----------
    v : torch.Tensor
        Any tensor

    Returns
    -------
    torch.Tensor
        Transposed and conjugated tensor.
    """
    if not isinstance(v, torch.Tensor):
        raise TypeError('v must be a torch.Tensor.')
    return torch.conj(v).t()

def check_poles(poles: torch.Tensor):
    """
    Checks if the poles are inside the unit circle.

    Parameters
    ----------
    poles : torch.Tensor
        Poles of the Blaschke product.
        It must be a 1-dimensional torch.Tensor with complex elements.
        It must be inside the unit circle (edge exclusive).

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    if not isinstance(poles, torch.Tensor):
        raise TypeError('poles must be a torch.Tensor.')
    
    if not poles.is_complex():
        raise TypeError('poles must be complex numbers.')
    
    if poles.ndim != 1:
        raise ValueError('poles must be a 1-dimensional torch.Tensor.')
    
    if torch.max(torch.abs(poles)) >= 1:
        raise ValueError('poles must be inside the unit circle!')

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
    if not isinstance(v, torch.Tensor):
        raise TypeError('v must be a torch.Tensor.')

    if not v.dtype.is_floating_point:
        raise TypeError('v must have real elements.')

    if v.ndim != 1:
        raise ValueError('v must be a 1-dimensional torch.Tensor.')
    
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
    bo = torch.zeros(n + 1, 3, dtype=torch.int32)
    bo[0, :] = torch.tensor([0, -1, -1])
    bo[1, :] = torch.tensor([n, -1, -1])
    bo[2, :] = torch.tensor([n // 2, 0, n])


    watch = 2  # Column currently being watched
    fill = 3  # Where to fill the new values

    # Fill the matrix with the ordering
    while fill < n + 1:

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
    mpoles : torch.Tensor, dtype=torch.complex64
        Poles of the Blaschke product. Must be a 1D tensor.
    eps : float, optional
        Accuracy of the complex discretization on the unit disc (default: 1e-6).

    Returns
    -------
    torch.Tensor, dtype=torch.float64
        Non-equidistant complex discretization.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    from blaschke import arg_inv

    # Validate input parameters
    check_poles(mpoles)

    if not isinstance(eps, float):
        raise TypeError('Eps must be a float.')
    
    if eps <= 0:
        raise ValueError('Eps must be a positive number.')

    # Calculate the non-equidistant complex discretization
    m = len(mpoles)
    #reduce the upper bound by small amount to have valid input for arg_inv
    z = torch.linspace(-torch.pi, torch.pi - eps/1000, m + 1, dtype=torch.float64)
    t = arg_inv(mpoles, z, eps)

    return t



def discretize_dr(mpoles: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """
    Computes the non-equidistant real discretization on the unit disc that refers to the given poles.

    Parameters
    ----------
    mpoles : torch.Tensor, dtype=torch.complex64
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
    from blaschke import argdr_inv

    if torch.max(torch.abs(mpoles)) >= 1:
        raise ValueError("Poles must be inside the unit disc")

    mpoles = torch.cat((torch.tensor([0.0]), mpoles))
    m = mpoles.size(0)
    stepnum = 2*(m-1) + 1
    #array of numbers ranging from -(m-1)*pi to (m-1)*pi, with pi as distance between each number
    z = torch.linspace(-(m-1)*torch.pi, (m-1)*torch.pi, steps=stepnum, dtype=torch.float64)
    z = z / m
    t = argdr_inv(mpoles, z, eps)
    return t

def dotdc(F: torch.Tensor, G: torch.Tensor, poles: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Computes complex discrete dot product of two functions in H^2(ID).
    TODO: F and G are expected to have the same number of elements, AND at least 2.

    Parameters
    ----------
    F : torch.Tensor, dtype=torch.complex64
        Values of the first function (ID -> IC) on the unit disk. 1D tensor. Must have the same number of elements as 'G'.
    G : torch.Tensor, dtype=torch.complex64
        Values of the second function (ID -> IC) on the unit disk. 1D tensor. Must have the same number of elements as 'F'.
    poles : torch.Tensor, dtype=torch.complex64
        Poles of the rational system. Must be a 1D tensor with elements inside the unit circle.
    t : torch.Tensor, dtype=torch.float64
        Arguments for which to evaluate the dot product. 1D tensor.

    Returns
    -------
    torch.Tensor, dtype=torch.complex64
        Values of the complex dot product of 'F' and 'G' at 't'.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    # Validate input parameters
    if not isinstance(F, torch.Tensor):
        raise TypeError('F must be a torch.Tensor.')
    if F.ndim!=1:
        raise ValueError('F must be a 1-dimensional torch.Tensor.')
    if not F.is_complex():
        raise TypeError('F must have complex elements.')
    
    if not isinstance(G, torch.Tensor):
        raise TypeError('G must be a torch.Tensor.')
    if G.ndim!=1:
        raise ValueError('G must be a 1-dimensional torch.Tensor.')
    if not G.is_complex():
        raise TypeError('G must have complex elements.')

    if F.size(0) != G.size(0):
        raise ValueError('F and G must have the same length.')
    
    check_poles(poles)
    
    if not isinstance(t, torch.Tensor):
        raise TypeError('t must be a torch.Tensor.')
    if t.ndim!=1:
        raise ValueError('t must be a 1-dimensional torch.Tensor.')
    if not t.is_floating_point():
        raise TypeError('t must have real elements.')



    # Compute the kernel values
    kernel_vals = torch.zeros(t.size(0)-1, dtype=torch.complex64)
    for i in range(kernel_vals.size(0)):
        kernel_vals[i] = kernel(torch.exp(1j * t[i]), torch.exp(1j * t[i]), poles)

    # Compute the complex discrete dot product
    s = torch.sum(F[:-1] * torch.conj(G[:-1]) / kernel_vals)

    return s

def dotdr(F: torch.Tensor, G: torch.Tensor, mpoles: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Computes discrete real dot product of two functions in H^2(ID).
    TODO: F and G AND t are expected to have the same number of elements

    Parameters
    ----------
    F : torch.Tensor, dtype=torch.complex64
        Values of the first function (ID -> IR) on the unit disk. 1D tensor. Must have the same number of elements as 'G'.
    G : torch.Tensor, dtype=torch.complex64
        Values of the second function (ID -> IR) on the unit disk. 1D tensor. Must have the same number of elements as 'F'.
    mpoles : torch.Tensor, dtype=torch.complex64
        Poles of the rational system. Must be a 1D tensor with elements inside the unit circle.
    t : torch.Tensor, dtype=torch.float64
        Arguments for which to evaluate the dot product. 1D tensor.

    Returns
    -------
    torch.Tensor, dtype=torch.complex64
        Values of the real dot product of 'F' and 'G' at 't'.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(F, torch.Tensor):
        raise TypeError('F must be a torch.Tensor.')
    if F.ndim!=1:
        raise ValueError('F must be a 1-dimensional torch.Tensor.')
    if not F.is_complex():
        raise TypeError('F must have complex elements.')
    
    if not isinstance(G, torch.Tensor):
        raise TypeError('G must be a torch.Tensor.')
    if G.ndim!=1:
        raise ValueError('G must be a 1-dimensional torch.Tensor.')
    if not G.is_complex():
        raise TypeError('G must have complex elements.')

    
    
    check_poles(mpoles)
    
    if not isinstance(t, torch.Tensor):
        raise TypeError('t must be a torch.Tensor.')
    if t.ndim!=1:
        raise ValueError('t must be a 1-dimensional torch.Tensor.')
    if not t.is_floating_point():
        raise TypeError('t must have real elements.')

    if F.size(0) != G.size(0) or F.size(0) != t.size(0):
        raise ValueError('F, G, and t must have the same length.')
    
    # Prepend zero to the poles
    mpoles = torch.cat((torch.zeros(1), mpoles))
    # Compute the kernel values
    kernel_vals = torch.zeros(t.size(0), dtype=torch.complex64)
    for i in range(kernel_vals.size(0)):
        kernel_vals[i] = kernel(torch.exp(1j * t[i]), torch.exp(1j * t[i]), mpoles)

    # Compute the real discrete dot product
    s = torch.sum(F * torch.conj(G) / (2 * torch.real(kernel_vals) - 1))

    return s





def kernel(y:torch.Tensor,z:torch.Tensor,mpoles: torch.Tensor) -> torch.Tensor:
    """
    Computes the weight function of discrete dot product in H^2(D).

    Parameters
    ----------
    y : torch.Tensor, dtype=torch.complex64
        First argument. Only one element.
    z : torch.Tensor, dtype=torch.complex64
        Second argument. Only one element.
    mpoles : torch.Tensor, dtype=torch.complex64
        Poles of the rational system. Must be a 1D tensor.

    Returns
    -------
    torch.Tensor
        Value of the weight function at arguments "y" and "z".
    """
    # Validate input parameters
    if not isinstance(y, torch.Tensor):
        raise TypeError('y must be a torch.Tensor.')
    if y.ndim != 0:
        raise ValueError('y must be a single number.')
    if not y.is_complex():
        raise TypeError('y must be a complex number.')

    if not isinstance(z, torch.Tensor):
        raise TypeError('z must be a torch.Tensor.')
    if z.ndim != 0:
        raise ValueError('z must be a single number.')
    if not z.is_complex():
        raise TypeError('z must be a complex number.')

    check_poles(mpoles)

    r = torch.zeros_like(y, dtype=torch.complex64)
    m = len(mpoles)
    if torch.allclose(y, z):
        for k in range(m):
            alpha = torch.angle(mpoles[k])
            R = torch.abs(mpoles[k])
            t = torch.angle(z)
            r += __poisson(R, t - alpha)
    else:
        for i in range(m):
            r += __MT(i, mpoles, y) * torch.conj(__MT(i, mpoles, z))
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
    return (1-r**2)/(1-2*r*torch.cos(t)+r**2)

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
    r = torch.ones_like(z, dtype=torch.complex64)
    for k in range(n):
        r *= (z - mpoles[k]) / (1 - torch.conj(mpoles[k]) * z)
    r *= torch.sqrt(1 - torch.abs(mpoles[n]) ** 2) / (1 - torch.conj(mpoles[n]) * z)
    return r

def multiplicity(mpoles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the multiplicity of all elements of the vector 'mpoles'.

    Parameters
    ----------
    mpoles : torch.Tensor, dtype=torch.complex64
        Poles with arbitrary multiplicities. It must be a 1D tensor.

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
    check_poles(mpoles)
    poles = mpoles
    #the tensor of multiplicities of poles
    mult = torch.ones_like(mpoles, dtype = torch.int32) 
    #the tensor of poles that contains each pole only once
    spoles = torch.zeros_like(mpoles, dtype = torch.complex64) 
    spoles_index=0

    while poles.numel() > 0:
        #indexes of the poles that are equal to the first element in poles, including the first element
        ind = [0]

        for i in range (1,poles.numel()):
            #instead of exact comparison, torch.allclose is used to compare the poles
            if torch.allclose(poles[i], poles[0]):
                ind.append(i)
                mult[spoles_index] += 1
        spoles[spoles_index] = poles[0]

        #remove the poles that are equal to the first pole INCLUDING the first pole
        for i in range(len(ind)-1,-1,-1):#remove backwards to avoid index errors
            poles = torch.cat((poles[0:ind[i]],poles[ind[i]+1:]))
        spoles_index += 1
        
    
    spoles = spoles[0:spoles_index]
    mult = mult[0:spoles_index]
    return spoles, mult


def subsample(sample:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
    """
    Interpolate values between uniform sampling points using linear interpolation.
    
    Parameters
    ----------
    sample : torch.Tensor, dtype=torch.complex64
        A 1D tensor of (?) uniformly sampled values on (?) [-pi, pi). 
        
        TODO: This is likely not the case. in test.m, e.g. when calling biortdc_coeffs, the value "sig" is complex, and not seemingly uniform, and not within [-pi, pi)
        Despite this, it is still used as an input to this function.
    x : torch.Tensor, dtype=torch.float64
        The values at which interpolation is to be computed. 1D tensor.

    Returns
    -------
    torch.Tensor
        The interpolated values at the points specified by x.
    """

    # Validate input parameters
    if not isinstance(sample, torch.Tensor):
        raise TypeError('sample must be a torch.Tensor.')
    if sample.ndim != 1:
        raise ValueError('sample must be a 1-dimensional torch.Tensor.')
    if not sample.is_complex():
        raise TypeError('sample must have complex elements.')
    
    
    if not isinstance(x, torch.Tensor):
        raise TypeError('x must be a torch.Tensor.')
    if x.ndim != 1:
        raise ValueError('x must be a 1-dimensional torch.Tensor.')
    if not x.is_floating_point():
        raise TypeError('x must have real elements.')

    # Number of samples
    len = sample.size(0)

    # Create a tensor of sample points
    sx = torch.linspace(-torch.pi, torch.pi, len)

    # torchinterp1d is used for linear interpolation
    # https://github.com/aliutkus/torchinterp1d
    y = interp1d(sx, sample, x)
    y = y[0]
    return y



def coeff_conv(length:int, poles:torch.Tensor, coeffs:torch.Tensor, base1:str, base2:str) -> torch.Tensor:
    """
    Convert the coefficients between the continuous systems base1 and base2.
    NOTE: coeffs has to be the same length as poles

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling.
    poles : torch.Tensor, dtype=torch.complex64
        Poles of the continuous systems. Must be a 1D tensor. Must be inside the unit circle.
    coeffs : torch.Tensor, dtype=torch.complex64
        Coefficients with respect to the continuous system 'base1'. 1D tensor.
    base1 : str
        Type of the continuous system to be converted.
    base2 : str
        Type of the converted continuous system.

    Returns
    -------
    torch.Tensor, 
        Converted coefficients with respect to the system 'base2'.
    
    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    from mt_sys import mt_system
    from biort_sys import biort_system
    from rat_sys import lf_system, mlf_system

    # Validate input parameters
    if not isinstance(length, int):
        raise TypeError('length must be an integer.')
    if length < 2:
        raise ValueError('length must be an integer greater than or equal to 2.')
    
    check_poles(poles)

    if not isinstance(coeffs, torch.Tensor):
        raise TypeError('coeffs must be a torch.Tensor.')
    if coeffs.ndim != 1:
        raise ValueError('coeffs must be a 1-dimensional torch.Tensor.')
    if not coeffs.is_complex():
        raise TypeError('coeffs must have complex elements.')
    
    #coeffs has to be the same length as poles
    if coeffs.size(0) != poles.size(0):
        raise ValueError('coeffs must have the same length as poles.')
    
    if not isinstance(base1, str):
        raise TypeError('base1 must be a string.')
    if base1 not in ['lf', 'mlf', 'biort', 'mt']:
        raise ValueError('Invalid system type for base1! Choose from lf, mlf, biort, mt.')
    
    if not isinstance(base2, str):
        raise TypeError('base2 must be a string.')
    if base2 not in ['lf', 'mlf', 'biort', 'mt']:
        raise ValueError('Invalid system type for base2! Choose from lf, mlf, biort, mt.')
    
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

    F = torch.matmul(g1,conj_trans(g1)) / length
    G = torch.matmul(g1,conj_trans(g2)) / length
    
    co = torch.linalg.solve(G,F)
    co = torch.matmul(co, conj_trans(coeffs))
    co = conj_trans(co)
    return co

def coeffd_conv(poles: torch.Tensor, coeffs: torch.Tensor, base1: str, base2: str, eps: float = 1e-6) -> torch.Tensor:
    """
    Converts the coefficients between the discrete systems base1 and base2.
    NOTE: coeffs has to be the same length as poles
    Parameters
    ----------
    poles : torch.Tensor, dtype=torch.complex64
        Poles of the discrete systems (1-dimensional tensor). Must be inside the unit circle.
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
    torch.Tensor, dtype=torch.complex64
        Converted coefficients with respect to the system 'base2'.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    from rat_sys import mlfdc_system
    from biort_sys import biortdc_system
    from mt_sys import mtdc_system

    # Validate input parameters
    check_poles(poles)
    
    if not isinstance(coeffs, torch.Tensor):
        raise TypeError('coeffs must be a torch.Tensor.')
    if coeffs.ndim != 1:
        raise ValueError('coeffs must be a 1-dimensional torch.Tensor.')
    if not coeffs.is_complex():
        raise TypeError('coeffs must have complex elements.')
    
    #coeffs has to be the same length as poles
    if coeffs.size(0) != poles.size(0):
        raise ValueError('coeffs must have the same length as poles.')

    if not isinstance(base1, str):
        raise TypeError('base1 must be a string.')
    if base1 not in ['mlfdc', 'biortdc', 'mtdc']:
        raise ValueError('Invalid system type for base1! Choose from mlfdc, biortdc, mtdc.')
    if not isinstance(base2, str):
        raise TypeError('base2 must be a string.')
    if base2 not in ['mlfdc', 'biortdc', 'mtdc']:
        raise ValueError('Invalid system type for base2! Choose from mlfdc, biortdc, mtdc.')
    
    if not isinstance(eps, float):
        raise TypeError('eps must be a float.')
    if eps <= 0:
        raise ValueError('eps must be a positive float.')
    
    # Helper function
    def get_system(base, mpoles, eps):
        if base == 'mlfdc':
            return mlfdc_system(mpoles, eps)
        elif base == 'biortdc':
            return biortdc_system(mpoles, eps)
        elif base == 'mtdc':
            return mtdc_system(mpoles, eps)
        else:
            raise ValueError('Invalid system type! Choose from mlfdc, biortdc, mtdc.')

    # Generate systems based on 'base1' and 'base2'
    g1 = get_system(base1, poles, eps)
    g2 = get_system(base2, poles, eps)

    # Convert coefficients between systems
    F = g1 @ conj_trans(g1) / coeffs.size(0)
    G = g1 @ conj_trans(g2) / coeffs.size(0)

    co = torch.linalg.solve(G, F)
    co = co @ conj_trans(coeffs)
    co = conj_trans(co)

    return co

def periodize(v: torch.Tensor, alpha: float, draw: bool = False) -> torch.Tensor:
    """
    NOTE: This function is untested and may not work as intended.

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

def rshow(*args):
    """
    NOTE: This function is untested and may not work as intended.

    Visualizes the given function, system, or systems.

    Usage:
        rshow(s)
        rshow(s1, s2)

    Parameters
    ----------
    *args : complex torch.Tensor
        Complex matrices with rows as elements of a function system.
        A system with one element is just a plain function.

    Returns
    -------
    None
        Plots of the real and imaginary parts of the elements of the function system.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    if len(args) == 0:
        raise ValueError('Please provide at least one parameter!')
    elif len(args) == 1:
        if not isinstance(args[0], torch.Tensor):
            raise ValueError('The parameter should be a torch.Tensor!')
        s1 = args[0]
        n1, m1 = s1.shape
    elif len(args) == 2:
        if not isinstance(args[0], torch.Tensor) or not isinstance(args[1], torch.Tensor):
            raise ValueError('The parameters should be torch.Tensors!')
        s1, s2 = args
        n1, m1 = s1.shape
        n2, m2 = s2.shape
        if n1 != n2 or m1 != m2:
            raise ValueError('The matrices should be of equal size!')
    else:
        raise ValueError('Please provide one or two matrices!')

    x = torch.linspace(0, 2 * math.pi, m1 + 1)[:-1]

    for i in range(n1):
        if len(args) == 1:
            plt.subplot(n1, 1, i + 1)
            plt.plot(x, s1[i].real, 'r', x, s1[i].imag, 'b', linewidth=1)
        elif len(args) == 2:
            plt.subplot(n1, 2, 2 * i + 1)
            plt.plot(x, s1[i].real, 'r', x, s1[i].imag, 'b', linewidth=1)

            plt.subplot(n1, 2, 2 * i + 2)
            plt.plot(x, s2[i].real, 'r', x, s2[i].imag, 'b', linewidth=1)

    plt.show()
