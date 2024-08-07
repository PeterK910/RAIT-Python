import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def arg_der(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Derivatives of the argument function of a Blaschke product.

    Parameters
    ----------
    a : torch.Tensor, dtype=torch.complex64
        Parameters of the Blaschke product. Must be a 1-dimensional torch.Tensor.
    t : torch.Tensor, dtype=torch.float64
        Values in [-pi, pi), where the function values are needed. Must be a 1-dimensional torch.Tensor.

    Returns
    -------
    torch.Tensor, dtype=torch.float64
        The derivatives of the argument function at the points in t.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(a, torch.Tensor):
        raise TypeError('"a" must be a torch.Tensor.')
    
    if a.dtype != torch.complex64:
        raise TypeError('"a" must be a complex torch.Tensor.')
    
    if a.ndim != 1:
        raise ValueError('"a" must be a 1-dimensional torch.Tensor.')
    
    if torch.max(torch.abs(a)) >= 1:
        raise ValueError('Elements of "a" must be inside the unit circle!')
    

    if not isinstance(t, torch.Tensor):
        raise TypeError('"t" must be a torch.Tensor.')
    
    if t.dtype != torch.float64:
        raise TypeError('"t" must be a torch.Tensor with float64 dtype.')

    if t.ndim != 1:
        raise ValueError('"t" must be a 1-dimensional torch.Tensor.')
    
    if torch.min(t) < -torch.pi or torch.max(t) >= torch.pi:
        raise ValueError('Elements of "t" must be in [-pi, pi).')
    

    # Calculate derivatives
    bd = torch.zeros_like(t, dtype=torch.float64)
    
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
    TODO: is len=1 allowed?
    poles : torch.Tensor, dtype=torch.complex64
        Parameters of the Blaschke product. Must be a 1-dimensional torch.Tensor.

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
    from .util import check_poles
    # Validate input parameters
    if not isinstance(len, int):
        raise TypeError('"len" must be an integer.')
    if len <= 0:
        raise ValueError('"len" must be a positive integer.')
    check_poles(poles)

    # Calculate the sampled Blaschke function
    t = torch.linspace(-torch.pi, torch.pi, len + 1)
    z = torch.exp(1j * t)
    b = torch.ones(len + 1, dtype=torch.complex64)
    for p in poles:
        b = b * (z - p) / (1 - torch.conj(p) * z)

    return b

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

def arg_fun(a: torch.Tensor, t: torch.Tensor, debug:bool = False) -> torch.Tensor:
    """
    Calculate the values of the argument function of a Blaschke product.

    Parameters
    ----------
    a : torch.Tensor, dtype=torch.complex64
        Parameters of the Blaschke product. Must be a 1-dimensional torch.Tensor.
    t : torch.Tensor, dtype=torch.float64
        REAL (not complex!) values where the function values are needed. Must be a 1-dimensional torch.Tensor.

    Returns
    -------
    torch.Tensor, dtype=torch.float64
        The values of the argument function at the points in t.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    from util import check_poles
    # Validate input parameters
    check_poles(a)
    

    if not isinstance(t, torch.Tensor):
        raise TypeError('"t" must be a torch.Tensor.')
    
    if t.dtype != torch.float64:
        raise TypeError('"t" must be a torch.Tensor with float64 dtype.')

    if t.ndim != 1:
        raise ValueError('"t" must be a 1-dimensional torch.Tensor.')

    # Calculate the argument function values
    b = torch.zeros(len(t), dtype=torch.float64)
    for i in range(len(a)):
        b += __arg_fun_one(a[i], t, debug)
        if debug:
            print(f"b = {b}, dtype = {b.dtype}")
    if debug:
        print(f"b before division = {b}")
    b /= len(a)
    if debug:
        print(f"b after division = {b}")
    return b

def __arg_fun_one(a: torch.Tensor, t: torch.Tensor, debug:bool = False) -> torch.Tensor:
    r = torch.abs(a)
    fi = torch.angle(a)
    mu = (1 + r) / (1 - r)
    
    gamma = 2 * torch.atan((1 / mu) * torch.tan(fi / 2))
    if debug:
        print(f"r = {r},fi={fi},mu = {mu},gamma = {gamma}")
    

    b = 2 * torch.atan(mu * torch.tan((t - fi) / 2)) + gamma
    if debug:
        print(f"b1 = {b}, type = {b.dtype}")
    #don't ever use fmod again, it's not working as expected
    b = (b + torch.pi) % (2 * torch.pi) - torch.pi  # move it in [-pi, pi)
    if debug:
        print(f"b2 = {b}, type = {b.dtype}")
    return b

def argdr_fun(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the values of the argument function of a Blaschke product.
    It is the same as arg_fun, but the output is continuous on IR.

    Parameters
    ----------
    a : torch.Tensor, dtype=torch.complex64
        Parameters of the Blaschke product. Must be a 1-dimensional torch.Tensor.
    t : torch.Tensor, dtype=torch.float64
        REAL (not complex!) values where the function values are needed. Must be a 1-dimensional torch.Tensor.

    Returns
    -------
    torch.Tensor, dtype=torch.float64
        The values of the argument function at the points in t.
        Continuous on IR.
    """
    from util import check_poles
    # Validate input parameters
    
    check_poles(a)

    if not isinstance(t, torch.Tensor):
        raise TypeError('"t" must be a torch.Tensor.')
    
    if t.dtype != torch.float64:
        raise TypeError('"t" must be a torch.Tensor with float64 dtype.')

    if t.ndim != 1:
        raise ValueError('"t" must be a 1-dimensional torch.Tensor.')
    
    # Initialize the result tensor
    b = torch.zeros(len(t), dtype=torch.float64)
    
    # Calculate the continuous argument function
    for j in range(t.size(0)):
        for i in range(a.size(0)):
            #wrapping parameters so that ndim is 1 instead of 0 for arg_fun
            tmp_a = torch.tensor([a[i]])
            tmp_t = torch.tensor([t[j]])
            bs = arg_fun(tmp_a, tmp_t)

            #unwrapping the result
            bs = bs[0]

            b[j] += bs + 2 * torch.pi * torch.floor((t[j] + torch.pi) / (2 * torch.pi))
    
    return b

def arg_inv(a: torch.Tensor, b: torch.Tensor, epsi: float = 1e-4) -> torch.Tensor:
    """
    Inverse images by the argument function of a Blaschke product.

    Parameters
    ----------
    a : torch.Tensor, dtype=torch.complex64
        Parameters of the Blaschke product. Must be a 1-dimensional torch.Tensor.
    b : torch.Tensor, dtype=torch.float64
        Values in [-pi, pi) whose inverse image is needed.
    epsi : float, optional
        Required precision for the inverses (default is 1e-4).

    Returns
    -------
    torch.Tensor
        Inverse images by the argument function of the points in 'b'.
    """

    print("arg_inv")
    from util import check_poles
    # Validate input parameters
    check_poles(a)
    
    if not isinstance(b, torch.Tensor):
        raise TypeError('"b" must be a torch.Tensor.')
    if b.dtype != torch.float64:
        raise TypeError('"b" must be a torch.Tensor with float64 dtype.')
    if b.ndim != 1:
        raise ValueError('"b" must be a 1-dimensional torch.Tensor.')
    if b.min() < -torch.pi or b.max() >= torch.pi:
        raise ValueError('Elements of "b" must be in [-pi, pi).')
    
    if not isinstance(epsi, float):
        raise TypeError('"epsi" must be a float.')
    if epsi <= 0:
        raise ValueError('"epsi" must be a positive float.')
    
    # Calculate inverse images
    if len(a) == 1:
        t = __arg_inv_one(a, b, epsi)
    else:
        t = __arg_inv_all(a, b, epsi)
    return t

def __arg_inv_one(a: torch.Tensor, b: torch.Tensor, epsi: float) -> torch.Tensor:
    """
    Calculate the inverse images by the argument function of a Blaschke product.

    Parameters
    ----------
    a : torch.Tensor
        Parameters of the Blaschke product.
    b : torch.Tensor
        Values in [-pi, pi) whose inverse image is needed.
    epsi : float
        Used to handle edge case of -pi not being exactly -pi.

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
    #handle edge case of -pi not being exactly -pi, by adding a very small number
    t = (t + torch.pi + epsi/1000) % (2 * torch.pi) - torch.pi  # Move it in [-pi, pi)
    
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
    print("arg_inv_all")
    torch.set_printoptions(precision=6)

    from util import bisection_order

    # Validate input parameters
    if not isinstance(a, torch.Tensor) or a.ndim != 1:
        raise ValueError('a must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(b, torch.Tensor) or b.ndim != 1:
        raise ValueError('b must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(epsi, float) or epsi <= 0:
        raise ValueError('epsi must be a positive float.')
    
    # Initialize variables
    n = len(b)
    s = bisection_order(n)
    x = torch.zeros(n+1, dtype=torch.float64)
    debug=False
    for i in range(n+1):
        
        if i == 0:
            v1, v2 = -torch.pi, torch.pi
            fv1, fv2 = -torch.pi, torch.pi # fv1 <= y, fv2 >= y

        elif i == 1:
            x[n] = x[0] + 2 * torch.pi
            continue
        else:
            v1 = x[s[i, 1]]
            v2 = x[s[i, 2]]
            print(f"i = {i}")
            print(f"x = {x}")
            print(f"s = {s}")
            print(f"v1 = x[s[{i},1]] = {v1}, v2 = x[s[{i},2]] = {v2}")

            #convert v1 and v2 to a format that argdr_fun can accept
            v1 = torch.tensor([v1], dtype=torch.float64)
            v2 = torch.tensor([v2], dtype=torch.float64)

            fv1, fv2 = arg_fun(a, v1), arg_fun(a, v2)

            #unwrapping the result
            fv1, fv2 = fv1[0], fv2[0]
        #i=0 ok
        if i > 0:
            print(f"i = {i}")
            print(f"b = {b}")
            print(f"s = {s}")
            print(f"ba for this round (b[s[{i},0]]): {b[s[i, 0]]}")
            debug=True

        ba = b[s[i, 0]]
        if fv1 == ba:
            x[s[i, 0]] = v1
            continue
        elif fv2 == ba:
            x[s[i, 0]] = v2
            continue
        else:
            xa = (v1 + v2) / 2

            #convert xa to a format that argdr_fun can accept
            xa = torch.tensor([xa], dtype=torch.float64)
            if i > 0:
                print(f"before while loop, calling arg_fun with a={a} \n and xa = {xa} = (v1 + v2) / 2 = \n ({v1} + {v2}) / 2")
            if(debug):
                fvk = arg_fun(a, xa, debug)
            else:
                fvk = arg_fun(a, xa)
            #fvk = arg_fun(a, xa)
            #unwrapping the result
            fvk = fvk[0]
            
            print(f"before while loop, fvk = {fvk}, ba = {ba}")
            print(f"while loop starts")
            j = 0
            error_count = 0
            while torch.abs(fvk - ba) > epsi:
                #delete later if not needed
                #TODO: find a condition that detects the case of fvk being way too far from ba, v1, v2 - which of the three is the best to check?
                """
                if not (v1 < fvk and fvk < v2):
                    raise ValueError(f"fvk is not in the interval [v1, v2]! fvk = {fvk}, v1 = {v1}, v2 = {v2}")
                """
                if(torch.abs(fvk - ba) > 6):
                    print(f"ALERT! fvk is too far from ba, fvk = {fvk}, ba = {ba}")
                    raise ValueError(f"fvk is too far from ba, fvk = {fvk}, ba = {ba}")
                if j == 100:
                    print(f"ALERT! j > 100, j = {j}")
                if j > 1000:
                    print(f"total times new difference was greater than old difference: {error_count}")
                    raise ValueError(f"j > 1000 and fvk still not near ba, j = {j}, fvk = {fvk}, ba = {ba}")
                print(f"{j}-eth iteration")
                j+=1
                print(f"fvk = {fvk}, ba = {ba}")
                if fvk == ba:
                    print(f"exact match for fvk and ba, x[s[{i},0]] = {xa}")
                    x[s[i, 0]] = xa
                    return x
                elif fvk < ba:
                    print(f"fvk < ba, v1 = xa")
                    v1 = xa
                else:
                    print(f"fvk > ba, v2 = xa")
                    v2 = xa
                print(f"v1 = {v1}, v2 = {v2}")
                if v1 > v2:
                    raise ValueError(f"v1 > v2 violates invariant, v1 = {v1}, v2 = {v2}")
                xa = (v1 + v2) / 2
                print(f"xa before arg_fun = {xa}")

                #convert xa to a format that argdr_fun can accept
                xa = torch.tensor([xa], dtype=torch.float64)
                print(f"calling arg_fun with a={a}, xa = {xa}")
                tmpfvk = fvk
                if(debug):
                    fvk = arg_fun(a, xa, debug)
                else:
                    fvk = arg_fun(a, xa)
                #fvk = arg_fun(a, xa)

                #unwrapping the result
                fvk = fvk[0]

                print(f"fvk after arg_fun = {fvk}, before, it was {tmpfvk}")
                old_dif = torch.abs(tmpfvk - ba)
                new_dif = torch.abs(fvk - ba)
                if new_dif > old_dif:
                    error_count += 1
                    print(f"ALERT! new difference is greater than old difference, old = {old_dif}, new = {new_dif}")
                    if(error_count > 10):
                        raise ValueError(f"new difference is greater than old difference, old = {old_dif}, new = {new_dif}")

            print(f"loop ends after {j} iterations, error count = {error_count}")
            x[s[i, 0]] = xa
    #drop the last element
    return x[:n]

def argdr_inv(a: torch.Tensor, b: torch.Tensor, epsi: float = 1e-4) -> torch.Tensor:
    """
    Inverse images by the argument function of a Blaschke product.

    Parameters
    ----------
    a : torch.Tensor, dtype=torch.complex64
        Parameters of the Blaschke product. Must be a 1-dimensional torch.Tensor.
    b : torch.Tensor, dtype=torch.float64
        Values in [-pi,pi) whose inverse image is needed. Must be a 1-dimensional torch.Tensor.
    epsi : float, optional
        Required precision for the inverses (default is 1e-4).

    Returns
    -------
    torch.Tensor, dtype=torch.float64
        Inverse images by the argument function of the points in 'b'.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    
    """
    from util import check_poles
    # Validate input parameters
    check_poles(a)

    if not isinstance(b, torch.Tensor):
        raise TypeError('"b" must be a torch.Tensor.')
    if b.dtype != torch.float64:
        raise TypeError('"b" must be a torch.Tensor with float64 dtype.')
    if b.ndim != 1:
        raise ValueError('"b" must be a 1-dimensional torch.Tensor.')
    if b.min() < -torch.pi or b.max() >= torch.pi:
        raise ValueError('Elements of "b" must be in [-pi, pi).')
    
    if not isinstance(epsi, float):
        raise TypeError('"epsi" must be a float.')
    if epsi <= 0:
        raise ValueError('"epsi" must be a positive float.')
    
    # Calculate inverse images
    if len(a) == 1:
        t = __argdr_inv_one(a, b, epsi)
    else:
        t = __argdr_inv_all(a, b, epsi)
    return t

def __argdr_inv_one(a:torch.Tensor, b:torch.Tensor, epsi:float)->torch.Tensor:
    """
    Calculate the inverse images by the argument function of a Blaschke product.

    Parameters
    ----------
    a : torch.Tensor
        Parameters of the Blaschke product.
    b : torch.Tensor
        Values in [-pi, pi) whose inverse image is needed.
    epsi : float
        Used to handle edge case of -pi not being exactly -pi.

    Returns
    -------
    torch.Tensor
        Inverse images by the argument function of the points in 'b'.
    """
    r = torch.abs(a)
    fi = torch.angle(a)
    mu = (1 + r) / (1 - r)
    
    gamma = 2 * torch.atan((1 / mu) * torch.tan(fi / 2))
    t = 2 * torch.atan((1 / mu) * torch.tan((b - gamma) / 2)) + fi
    #handle edge case of -pi not being exactly -pi, by adding a very small number before modulo
    t = (t + torch.pi + epsi/1000) % (2 * torch.pi) - torch.pi  # move it in [-pi,pi)
    
    return t

def __argdr_inv_all(a:torch.Tensor, b:torch.Tensor, epsi:float)->torch.Tensor:
    """
    Inverse when the number of poles is greater than 1.
    Uses the bisection method with an enhanced order of calculation
    """
    from util import bisection_order

    n = len(b)
    s = bisection_order(n)
    x = torch.zeros(n+1, dtype=torch.float64)
    for i in range(n+1):
        if i == 0:
            v1 = -torch.pi
            v2 = torch.pi
            fv1 = -torch.pi  # fv1 <= y
            fv2 = torch.pi   # fv2 >= y
        elif i == 1:
            x[n] = x[0] + 2 * torch.pi  # x(s(1,1))
            continue
        else:          
            v1 = x[s[i, 1]]
            v2 = x[s[i, 2]]
            
            #convert v1 and v2 to a format that argdr_fun can accept
            v1 = torch.tensor([v1], dtype=torch.float64)
            v2 = torch.tensor([v2], dtype=torch.float64)

            fv1 = (argdr_fun(a, v1)-v1/2)/a.size(0)
            fv2 = (argdr_fun(a, v2)-v2/2)/a.size(0)

            #unwrapping the result
            fv1, fv2 = fv1[0], fv2[0]

        ba = b[s[i, 0]]
        if fv1 == ba:
            x[s[i, 0]] = v1
            continue
        elif fv2 == ba:
            x[s[i, 0]] = v2
            continue
        else:
            xa = (v1 + v2) / 2

            #convert xa to a format that argdr_fun can accept
            xa = torch.tensor([xa], dtype=torch.float64)

            fvk = (argdr_fun(a, xa)-xa/2)/a.size(0)

            #unwrapping the result
            fvk = fvk[0]

            while torch.abs(fvk - ba) > epsi:
                if fvk == ba:
                    x[s[i, 0]] = xa
                    break
                elif fvk < ba:
                    v1 = xa
                else:
                    v2 = xa
                xa = (v1 + v2) / 2

                #convert xa to a format that argdr_fun can accept
                xa = torch.tensor([xa], dtype=torch.float64)

                fvk = (argdr_fun(a, xa)-xa/2)/a.size(0)

                #unwrapping the result
                fvk = fvk[0]

            x[s[i, 0]] = xa
    #drop the last element
    return x[:n]

def arg_inv_anim(a: torch.Tensor, n: int) -> None:
    """
    Shows an animation related to the inverse of an equidistant discretization
    by an argument function.

    Parameters
    ----------
    a : torch.Tensor
        The parameter of a Blaschke function.
    n : int
        Number of discretization points.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    if not isinstance(a, torch.Tensor) or a.ndim != 1:
        raise ValueError("The parameter 'a' should be a 1-dimensional torch.Tensor!")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("The parameter 'n' should be a positive integer!")
    if torch.max(torch.abs(a)) >= 1:
        raise ValueError("The parameter 'a' should be inside the unit disc!")

    t = torch.linspace(-torch.pi, torch.pi, n + 1)[:-1]  # Discretization
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