import torch
import decimal

def output_to_file(arg:torch.Tensor):
    # Convert the tensor to a string
    torch.set_printoptions(precision=6)
    output = ""
    x,y = 1,1
    if arg.dim() == 0:
        x,y = 1,1
        if arg.dtype != torch.complex64:
            print(arg.item())
            output += "{:.6f}".format(arg.item()) + "\n"
        else:
            print(arg.real.item(), arg.imag.item())
            output += "{:.6f}".format(arg.real.item()) + "+" + "{:.6f}".format(arg.imag.item()) + "i\n"
    elif arg.dim() == 1:
        y = arg.size(0)
        for i in range(y):
            if arg[i].dtype != torch.complex64:
                print(arg[i].item())
                output += "{:.6f}".format(arg[i].item()) + ","
            else:
                print(arg[i].real.item(), arg[i].imag.item())
                output += "{:.6f}".format(arg[i].real.item()) + "+" + "{:.6f}".format(arg[i].imag.item()) + "i,"
        output = output[:-1] + "\n"
    elif arg.dim() == 2:
        y,x = arg.size(0), arg.size(1)
        for i in range(y):
            for j in range(x):
                if arg[i,j].dtype != torch.complex64:
                    print(arg[i,j].item())
                    output += "{:.6f}".format(arg[i,j].item()) + ","
                else:
                    print(arg[i,j].real.item(), arg[i,j].imag.item())
                    output += "{:.6f}".format(arg[i,j].real.item()) + "+" + "{:.6f}".format(arg[i,j].imag.item()) + "i,"
            output = output[:-1] + "\n"
    else:
        raise ValueError("Only 0, 1, or 2 dimensional tensors are supported.")
    
            
    write_to_file(output)
    
def write_to_file(output):
    # Open a file in write mode
    with open("output_python.txt", "w") as file:
        # Write the output to the file
        file.write(output)

    print("Output written to file.")