
import torch
import thunder.torch as ltorch
import thunder
from functools import partial, wraps
import inspect
'''
@wraps(torch.Tensor.is_cuda) is equivalent to partial(update_wrapper, wrapped=torch.Tensor.is_cuda)
'''
'''
@wraps(torch.Tensor.is_cuda)
def my_func(x: torch.Tensor) :

means

torch.Tensor.is_cuda(my_func)
'''

# def first_decorator(func): #same as @wrap(first_decorator)
#     def wrapper(x):
#         print(f"wrapper: {x}")
#         ret = func(x)
#         return ret
#     return wrapper

class myClass():
    def __init__(self):
        self.x = ""
    
    def add(self, x):
        """
        is this the doc?
        """
        print('myClass')
        return str(x)
    
    def subtract(self, x):
        return x - 10

@wraps(myClass.add)
def fn_with_wrap(x):
    print(f"fn_with_wrap: {x}")
    return x

def fn_no_wrap(x):
    print(f"fn_no_wrap: {x}")
    return x

def tmp_func(x):
    print(f"tmp function: {x}")
    return x


def my_decorator(func):#func = torch.Tensor.is_cuda
    @wraps(func)
    def wrapper(x: torch.Tensor):
        print('decorator')
        ret = func(x)
        return ret
    return wrapper


# def bigger_func():
#     def my_decorator(func):#func = torch.Tensor.is_cuda
#         @wraps(func)
#         def wrapper(x: torch.Tensor):
#             print('decorator')
#             ret = func(x)
#             return ret
#         return wrapper
    
#     my_decorator(torch.Tensor.is_cuda)

@wraps(torch.Tensor.is_cuda)
def my_func(x: torch.Tensor) -> bool:
    print((x.__dict__))
    return torch.abs(x)

@wraps(ltorch.is_cuda)
def my_func2(x: torch.Tensor) -> bool:
    print((x.__dict__))
    return torch.abs(x)

def my_func3(x: torch.Tensor) -> bool:
    print((x.__dict__))
    return torch.abs(x)

@my_decorator
def my_func4(x: torch.Tensor) -> bool:
    print((x.__dict__))
    return torch.abs(x)

jit_tmp = thunder.functional.jit(my_func)
# @my_decorator
# def my_func2(x):
#     print('hi')
#     return torch.abs(x)

x = torch.tensor([1, 1, 1, 0, 1])
inspect.signature()
#  my_func2(x)
# print('')

import numpy as np
np.abs