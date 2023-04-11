# Add a new reverse-mode automatic differentiation rule

Despite the "automatic" name in the approach of computing gradient of some loss
function, it's automatic only for users. For developers, it's a lot of work to
write the code that computes the gradient. Adding a new rule is not a trivial
task, and it's not always possible to do it without a deep understanding of the
internals of the library and testing infrastructure.

This document describes the process of adding a new rule for reverse-mode
automatic differentiation. It's a step-by-step guide that describes how to
implement a new rule, test it, and submit for review.

## Table of contents
1. [Introduction](#introduction)
2. [Deriving the reverse mode rule (vector-Jacobian product)](#deriving-the-reverse-mode-rule-vector-jacobian-product)
3. [Implementing the vector-Jacobian product](#implementing-the-vector-jacobian-product)
4. [Testing the vector-Jacobian product](#testing-the-vector-jacobian-product)
5. [Submitting a PR](#submitting-a-pr)

## Introduction
Some glossary first:
* _primitive/primal_ is a function that can be differentiated. It can be a function
    from the standard library, or a function from a library that has a custom
    differentiation rule.
* _reverse differentiation rule_ is a function that computes the
    vector-Jacobian products of the result of the primitive with respect to the
    arguments of the primitive.
* _forward differentiation rule_ is a function that computes the
    Jacobian-vector products of the arguments of the primitive with respect to
    the result of the primitive.

For a function `f` with vector-valued output of size `n` and vector-valued input
of size `m`, the Jacobian is a `n x m` matrix of partial derivatives of the
output with respect to the input. Forward-mode differentiation computes the
Jacobian-vector product, which takes a vector of size `m` and returns a vector
of size `n`. Reverse-mode differentiation computes the vector-Jacobian product
(or Jacobian-transpose-vector product), which takes a vector of size `n` and
returns a vector of size `m`.

JAX has a very nice
[tutorial](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
on automatic differentiation. It's a good read if you want to learn more about
the theory behind automatic differentiation and become familiar with the
functional API of JAX. Another good resource is the ChainRules.jl documentation,
which describes the forward and reverse mode rules for a number of Julia
primitives. Specifically, the
["Propagators page"](https://juliadiff.org/ChainRulesCore.jl/stable/maths/propagators.html),
which describes the pushforward and pullback functions, is a good place to start
for a mathematical intuition behind the forward and reverse mode rules. The page
on
[Deriving array rules](https://juliadiff.org/ChainRulesCore.jl/stable/maths/arrays.html) is also
a good read if you want to learn how to derive the rules for operations on
arrays beyond the element-wise operations. Finally, the
["Changing the Primal" page](https://juliadiff.org/ChainRulesCore.jl/stable/design/changing_the_primal.html)
describes why the reverse mode rules are implemented as augmented primal
functions (spoiler: to be able to save the information necessary for the reverse
mode).

Let's consider element-wise multiplication of two vectors as an example.

## Deriving the reverse mode rule (vector-Jacobian product)
Since the primitive is element-wise multiplication, the Jacobian is a diagonal
matrix and the Jacobian-vector product is simply the element-wise product of the
diagonal elements with the vector.

f(x, y) = x * y

∂fᵢ/∂xⱼ = δᵢⱼ * yᵢ

∂fᵢ/∂yⱼ = δᵢⱼ * xⱼ

δᵢⱼ is the Kronecker delta, which is 1 if i == j and 0 otherwise.

Let's implement the manual computation of the vector-Jacobian product for
element-wise multiplication of two vectors and compare it with the result of
PyTorch's autograd.
```python
import torch
torch.manual_seed(0)

x = torch.randn(3, requires_grad=True)
y = torch.randn(3, requires_grad=True)

z = x * y
v = torch.randn(3)
z.backward(v)

def manual_backward1(x, y, v):
    # Explicit construction of the Jacobian
    J_x = torch.diag(y)
    J_y = torch.diag(x)
    return v @ J_x, v @ J_y

def manual_backward2(x, y, v):
    # Element-wise product of the vector with the diagonal elements
    return v * y, v * x

print("x.grad, y.grad           ", (x.grad, y.grad))
print("manual_backward1(x, y, v)", manual_backward1(x.detach(), y.detach(), v))
print("manual_backward2(x, y, v)", manual_backward2(x.detach(), y.detach(), v))
# x.grad, y.grad            (tensor([ 0.2293, -0.9089,  1.0060]), tensor([ 0.6216, -0.2459,  1.5671]))
# manual_backward1(x, y, v) (tensor([ 0.2293, -0.9089,  1.0060]), tensor([ 0.6216, -0.2459,  1.5671]))
# manual_backward2(x, y, v) (tensor([ 0.2293, -0.9089,  1.0060]), tensor([ 0.6216, -0.2459,  1.5671]))
```

## Implementing the vector-Jacobian product
Once we have the formula for the vector-Jacobian product, we can implement it in
code. In Thunder, all reverse differentiation rules are registered in the
`augmented_forward_impls` and `backward_impls` dictionaries in
`thunder/core/transforms.py`. The keys of the dictionary are the enums of the
primitives, and the values are the augmented primal functions that compute the
primitive result, save necessary information for the reverse mode rule for the
`augmented_forward_impls`, and a function that computes the vector-Jacobian
product given the vector and the saved from the primal computation for the
`backward_impls`.

```python
# continue from the previous code block
from thunder.core.transforms import augmented_forward_impls, register_augmented_forward, register_backward
from thunder.core.prims import Ops

# Remove the existing rule for element-wise multiplication
try:
    del augmented_forward_impls[Ops.MUL]
except KeyError:
    pass

# The decorator registers the augmented primal function for the given primitive
@register_augmented_forward(Ops.MUL)
def augmented_mul(x, y):
    """Augmented version of the element-wise multiplication primitive.

    Returns:
        The result of the element-wise multiplication, saved information for
        the reverse mode rule, and a callable that computes the vector-Jacobian
        product given the vector and the saved information.
    """
    saved_info = (x, y)
    return x * y, saved_info

@register_backward(Ops.MUL)
def mul_backward(x, y, v):
    """Computes the vector-Jacobian product given the vector and the saved
    information.
    """
    # Note that this function is exactly the same as the manual_backward2
    return v * y, v * x

# Test the new rule
def func(x, y):
    return x * y

from thunder.core.transforms import vjp
from thunder import make_traced
# vjp_traced is a callable that computes the vector-Jacobian product given the
# primal inputs and the vector to multiply with the Jacobian
# it returns a tuple of the primal result and the result of the vector-Jacobian product
vjp_traced = make_traced(vjp(func), executor="torch")

print("x.grad, y.grad", (x.grad, y.grad))
# 2nd element of the tuple is the vector-Jacobian product
print("Thunder's vjp ", vjp_traced((x, y), (v,))[1])
# x.grad, y.grad (tensor([ 0.2293, -0.9089,  1.0060]), tensor([ 0.6216, -0.2459,  1.5671]))
# Thunder's vjp  (tensor([ 0.2293, -0.9089,  1.0060]), tensor([ 0.6216, -0.2459,  1.5671]))
```

Since this particular rule is so simple, we can also implement it using the less
readable two lines of code.

```python
register_augmented_forward(Ops.MUL)(lambda x, y: (x * y, (x, y), ))
register_backward(Ops.MUL)(lambda x, y, v: (v * y, v * x))
```

## Testing the vector-Jacobian product
Currently tests are implemented using OpInfos and the tests are located in
`test_grad.py`. The tests are parametrized by the OpInfo and the executor
("torch" or "nvfuser"). The tests are run for all the OpInfos that have
`float64` as a supported dtype. The tests are run for all the executors that
support the OpInfo. For each OpInfo, the test suite first checks the `augmented_forward_impls`
dictionary to see if there is a registered rule for the OpInfo. If there is no
rule, the test is not generated. If there is a rule, the test suite generates a
test for each of the OpInfo's sample inputs. The test checks that the result of
the vector-Jacobian product is correct by verifying some mathematical properties
of the vector-Jacobian product and compared against the result of numerical
Jacobian-vector product. Currently, we don't have debugging tools for printing
the whole Jacobian matrices discrepancies similar to
`torch.autograd.gradcheck(fast_mode=False)`.

Once a new rule is added, the test suite can be run to verify that the rule is
correct:

```bash
python -m pytest tests/test_grad.py -k "_mul_" -vvv
```

## Submitting a PR
Checklist for submitting a PR:
* If OpInfo does not exist, add it to `opinfos.py`, covering representative
    input cases.
* Add the rule to `augmented_forward_impls` and `backward_impls` in
  `thunder/core/transforms.py`.
* Run the test suite to verify that the OpInfo-generated tests pass.
