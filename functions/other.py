import torch

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
    if n == 1:
        return torch.tensor([0])
    elif n == 2:
        return torch.tensor([1, 0])
    else:
        return torch.cat((bisection_order(n//2)*2, bisection_order(n//2)*2+1))


