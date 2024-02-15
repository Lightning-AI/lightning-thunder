.. module:: thunder.clang

thunder.clang
=============

Thunder Core Language


.. autosummary::
    :toctree: generated/

    maybe_convert_to_dtype
    device_put
    arange
    convolution
    full
    full_like
    uniform
    uniform_like
    diagonal
    expand
    flatten
    movedim
    reshape
    slice_in_dim
    squeeze
    transpose
    stride_order
    take
    index_add
    take_along_axis
    scatter_add
    unsqueeze
    cat
    stack
    compute_broadcast_shape
    matrix_transpose
    maybe_broadcast

Unary
~~~~~

.. autosummary::
    :toctree: generated/

    abs
    acos
    acosh
    asin
    asinh
    atan
    atanh
    bitwise_not
    ceil
    cos
    cosh
    erf
    erfc
    erfcinv
    erfinv
    exp
    exp2
    expm1
    floor
    isfinite
    lgamma
    log
    log10
    log1p
    log2
    ndtri
    neg
    reciprocal
    round
    rsqrt
    sigmoid
    sign
    signbit
    silu
    sin
    sinh
    sqrt
    tan
    tanh
    trunc

Binary
~~~~~~

.. autosummary::
    :toctree: generated/

    add
    atan2
    bitwise_and
    bitwise_or
    bitwise_xor
    copysign
    eq
    floor_divide
    fmod
    mod
    ge
    gt
    logical_and
    le
    lt
    mul
    ne
    nextafter
    pow
    remainder
    sub
    true_divide

Conditional
~~~~~~~~~~~

.. autosummary::
    :toctree: generated/

    where
