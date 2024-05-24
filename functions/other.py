import torch
import math

"""
Calculates the imaginary part of v using FFT

:param v: input tensor with real elements
:type v: tensor
:returns: tensor with imaginary part of v
:rtype: tensor
"""
def addimag(v: torch.Tensor) -> torch.Tensor:
    # Check if input is a tensor
    if (not torch.is_tensor(v)):
        raise TypeError("Input must be a tensor")
    # Check if tensor has 1 row
    if v.size(dim=0) != 1:
        raise ValueError("Input tensor must have exactly 1 row")
    # Check if tensor has only real elements
    if not torch.is_floating_point(v):
        raise ValueError("Input tensor must have real elements")
    
    # Calculate imaginary part of v using FFT
    vf = torch.fft.fft(v)
    vif = __mt_arrange(vf)
    vi = torch.fft.ifft(vif)
    return vi

"""
Rearrage FFT(v) so that lots of zeros appear on the right side of the FFT
"""
def __mt_arrange(t: torch.Tensor) -> torch.Tensor:
    mt = t.size(dim=1)
    ta = torch.zeros(t.size())
    ta[0] = t[0]
    for i in range(1, mt//2 + 1): # 1 to mt//2 inclusive
        ta[i] = t[i] + torch.conj(t[mt+1-i])
        ta[mt+1-i] = 0
    return ta

"""
Gives a better order for multiple bisection runs

:param n: number of points
:type n: int
:returns: a 3-by-(n+1) tensor with the order of calculation
:rtype: tensor
"""
def bisection_order(n: int) -> torch.Tensor:
    bo = torch.zeros(n+1, 3)
    bo[0] = torch.tensor([0, -1, -1])
    bo[1] = torch.tensor([n, -1, -1])
    bo[2] = torch.tensor([math.floor(n/2), 0, n])

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
Converts the coefficients coeffs between the continuous systems base1 and base2
Requires the implementation of lf_system, mlf_system, biort_system, and mt_system
"""
def coeff_conv(len:int, poles, coeffs, base1, base2):
    pass
"""
Converts the coefficients coeffs between the discrete systems base1 and base2.
Requires the implementation of mlfdc_system, biortdc_system, and mtdc_system
"""
def coeffd_conv(poles, coeffs, base1, base2, eps):
    pass

"""
Computes the non-equidistant complex discretization on the unit disc that refers to the given poles.
Requires the implementation of arg_inv
"""
def discretize_dc(mpoles, eps):
    pass

"""
Computes the non-equidistant real discretization on the unit disc that refers to the given poles.
Requires the implementation of argdr_inv
"""
def discretize_dr(mpoles, eps):
    pass
"""
Computes complex discrete dot product of two function in H^2(ID).  
:param F: ID-->IC, first analytic function on the disk unit
:type F: function
:param G: ID-->IC, second analytic function on the disk unit
:type G: function
:param poles: poles of the rational system
:type poles: ??
:param t: arguments(s)
:tpye t: ??

:returns: values of the complex dot product of "F" and "G" at "t"
Requires the implementation of kernel
"""
def dotdc(F,G,poles,t):
    pass

def dotdr():
    pass

