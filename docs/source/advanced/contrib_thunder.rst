Contributing to Thunder
#######################

We appreciate your feedback and contributions.
If you have feature requests, questions, or want to contribute code or config files,
please don't hesitate to use the `GitHub Issue tracker <https://github.com/Lightning-AI/lightning-thunder/issues>`_.

We welcome all individual contributors, regardless of their level of experience or hardware.
Your contributions are valuable, and we are excited to see what you can accomplish in this collaborative and supportive environment.

For a simple general overview of *Thunder*, we recommend reading
:doc:`inside Thunder <../advanced/inside_thunder>` first.


================
Adding operators
================
Adding operators might be one of the easiest and fun ways to get involved in contributing to *Thunder*.
The `operator GitHub Issue tracker <https://github.com/Lightning-AI/lightning-thunder/issues?q=is%3Aissue+is%3Aopen+label%3Aoperators>`_
provides a great starting point in deciding which operation to work on first.

The subsections below are structured as follows

* `Primitives`_, `The Core Language`_, `The Torch Language`_, `Language Context`_
  describe the hierarchy of operations and abstractions around them in *Thunder* in general terms.
* `Adding operations to the Torch executor`_ moves from *theory* to *practice* where we inspect real contributors' pull requests.

We recommend reading the document **sequentially**!

----------
Primitives
----------
The *lowest* level is the primitive operations, defined in `thunder/core/prims.py <https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/core/prims.py>`_.
Primitive operations, as seen in the `Representing operations <../advanced/inside_thunder.rst#representing-operations>`__ section,
describe all the computation performed, and they are intended to be as simple as possible so that
executors like `nvFuser <https://github.com/NVIDIA/Fuser>`_ find it easy to manipulate them.
*Thunder's* primitives are similar to `PyTorch’s <https://pytorch.org>`_
`primTorch primitives <https://github.com/pytorch/pytorch/tree/main/torch/_prims>`_,
and are based on `JAX’s <https://github.com/google/jax>`_
`jax.lax operations <https://jax.readthedocs.io/en/latest/jax.lax.html>`_.

Primitives have several parts, as defined by the ``make_prim`` function.
Most importantly they have an ``id``, a ``name``, and a ``meta`` function.
The meta function performs error checking and maps the metadata (like ``dtype``, ``shape``, ``device``) of inputs to the primitive metadata of outputs.
For operations that are part of a class, like the elementwise unary or reduction operations, they often share a common meta function.
More unique operations, like ``slice``, define their own
`meta functions <https://github.com/Lightning-AI/lightning-thunder/blob/888b46324462fba70f93d5017bc0d99025f05091/thunder/core/prims.py#L2812>`_.

The actual execution of primitive operations is handled by executors like
`nvFuser <https://github.com/NVIDIA/Fuser>`_ or
`PyTorch <https://pytorch.org>`_ – more on that in a moment.

Before adding a primitive, check with the team on its design.
It might be appropriate to add primitives when necessary to describe the semantics of an operation or to improve the numerical accuracy or speed of operations.

There is a tradeoff with the design of primitive operations one has to keep in mind.
On one hand, fewer primitive operations can make program transformation, and execution easier.
Fewer primitives means fewer transformation rules – since transformation rules are defined on primitives – and a smaller interface with executors.
On the other hand, too few primitive operations may make it hard, or impossible, to express all the operations that users are interested in.
Too few primitive operations may also make it difficult to execute programs quickly and numerically accurately.

For example, the ``expm1`` operation can mathematically be defined in terms of the ``exp`` and subtraction operations,
and so it does not need to be a primitive to enable any functionality.
Many libraries, including `C++’s standard library <https://en.cppreference.com/w/c/numeric/math/expm1>`_,
still define an ``expm1`` operation for numerical accuracy, and so ``expm1`` is a
`primitive <https://github.com/Lightning-AI/lightning-thunder/blob/888b46324462fba70f93d5017bc0d99025f05091/thunder/core/prims.py#L1791>`_ in *Thunder*.


-----------------
The Core Language
-----------------
Above the primitives is the ``core`` language, or `clang <https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/clang/__init__.py>`_.
Clang operations are mostly written like any other Python operation.
They ultimately call the primitive operations, although they may call other operations before doing so
(for example, ``clang.foo`` might call ``clang.bar`` which calls ``prims.bar``).

Core language operations are intended to be common functionality that’s useful when
defining user-facing languages like ``torch`` or ``numpy``.
Many of these operations are just wrappers around primitive operations.
For example, the elementwise binary primitives are as simple as possible, so they don’t perform broadcasting or type promotion.
The core language elementwise binary operations, however, do perform broadcasting and type promotion.
For example, take a look at the following implementation of ``add`` from `clang <https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/clang/__init__.py>`_

.. code-block:: python
  :lineno-start: 1602

  def _elementwise_binary_wrapper(a, b, *, prim, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT):
      computation_dtype, result_dtype = utils.elementwise_type_promotion(a, b, type_promotion_kind=type_promotion_kind)

      a, b = maybe_broadcast(a, b)
      a, b = maybe_convert_to_dtype(a, computation_dtype), maybe_convert_to_dtype(b, computation_dtype)

      result = prim(a, b)
      result = maybe_convert_to_dtype(result, result_dtype)

      return result


  @clangop(method_name="add")
  def add(a, b):
      return _elementwise_binary_wrapper(a, b, prim=prims.add)


Before adding a core language operation consider if the functionality expressed is universal enough.

As a style note, operations in *Thunder* should defer as much error checking as possible.
For example, if a primitive’s meta function will perform an error check for ``X``,
then the core language operation that calls it should generally not also check for ``X``.


------------------
The Torch Language
------------------
To translate ``torch`` operations into something that *Thunder* understands we define a
`torch language <https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/torch/__init__.py>`_.
Operations in the ``torch`` should reflect the behavior of their corresponding torch operations (small deviations are sometimes OK).

When a program is interpreted, torch operations are remapped into these operations, which ultimately call primitive operations.


----------------
Language Context
----------------
In the ``core`` and ``torch`` languages functions are decorated to set a *language context* and – for ``torch`` operations – to describe
how to map operations like ``torch.foo`` into ``thunder.torch.foo``.

The language context determines what properties and methods tensor objects have.
For example, when ``a + b`` is written and the first argument is an array or tensor
(so, `TensorProxy.__add__ is invoked <https://github.com/Lightning-AI/lightning-thunder/blob/71fc8cda4e42b818372b6e9f1c99c9cc3a5c2e38/thunder/core/proxies.py#L1310>`_),
the language context decides what that addition
`means <https://github.com/Lightning-AI/lightning-thunder/blob/71fc8cda4e42b818372b6e9f1c99c9cc3a5c2e38/thunder/core/langctxs.py#L66>`_.
Or when ``a.size`` is used, the language context determines what that means (and it’s different in PyTorch and NumPy).


---------------------------------------
Adding operations to the Torch executor
---------------------------------------
Now that we are familiar with the hierarchy of operations and the underlying language contexts, let's see some
examples of adding operations.

For simplicity, we only cover adding operations to the ``torch`` executor.
The sections below are meant to be read **sequentially**.

~~~~~~~~~~~~~~~~~~
Adding a primitive
~~~~~~~~~~~~~~~~~~
A good example of adding a primitive operation to the ``torch`` executor is
the `PR #136 <https://github.com/Lightning-AI/lightning-thunder/pull/136>`_
which adds support for `torch.Tensor.unfold <https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html>`_.

Let's outline some of its key parts.

Consider the following update to
`thunder/core/prims.py <https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/core/prims.py>`_

.. code-block:: python
   :emphasize-lines: 5
   :lineno-start: 152

   SLICE = auto()
   SQUEEZE = auto()
   TRANSPOSE = auto()
   UNFOLD = auto()
   VIEW = auto()
   # Memory layout prims (Experimental)
   STRIDE_ORDER = auto()

.. code-block:: python
   :lineno-start: 3082

   def unfold_meta(a: TensorProxy, /, dim: int, size: int, step: int) -> TensorProxy:
       dim = utils.canonicalize_dim(a.ndim, dim)
       max_size = 1 if a.ndim == 0 else a.shape[dim]

       utils.check(
       size <= max_size, lambda: f"Maximum size for tensor at dimension {dim} is {max_size} but size is {size}"
       )
       utils.check(size >= 0, lambda: f"Size is {size} but must be >= 0")
       utils.check(step > 0, lambda: f"Step is {step} but must be > 0")

       shape = list(a.shape)
       shape.append(size)
       shape[dim] = (shape[dim] - size) // step + 1

       return TensorProxy(like=a, shape=shape)


    unfold = make_prim(PrimIDs.UNFOLD, "unfold", meta=unfold_meta, tags=(OpTags.SHAPE_OP,))

The above registers a primitive symbol ``unfold`` using ``make_prim`` with ``id=PrimIDs.UNFOLD``,
``name=unfold``, and ``meta=unfold_meta``. One can see that ``unfold_meta`` follows the signature
of the underlying ``torch.Tensor.unfold`` operation
(so that the primitive is directly modeled after the PyTorch operation)
with the only exception of expecting a ``TensorProxy``
and not the ``torch.Tensor`` as its input. The rest of the function checks the inputs and returns
a ``TensorProxy`` of the appropriate shape. ``like=a`` means that the output will inherit the meta-data
like ``device`` and ``dtype`` from ``a``. The primitive is also tagged with ``tags=(OpTags.SHAPE_OP,)``,
and, therefore, is associated with shape-based operations.
We use tags to additionally group operations for group-specific operation optimizations inside *Thunder*.

Once the symbol is created, we need to tell *Thunder* how to *execute* it.
Since we are updating the ``torch`` executor, the following lines are added to the
`executors/torchex.py <https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/executors/torchex.py>`_ file

.. code-block:: python
   :emphasize-lines: 2
   :lineno-start: 465

   unbind = _register_torch_operation("unbind")
   unfold = _register_torch_operation("unfold", module=torch.Tensor)
   unsqueeze = _register_torch_operation("unsqueeze")

.. code-block:: python
   :emphasize-lines: 2
   :lineno-start: 536

   _register_implementation(prims.transpose, checker=_always_executable, execution_transform=_transpose_prim_transform)
   _register_implementation(prims.unfold, unfold, checker=_always_executable)
   _register_implementation(prims.view, view, checker=_always_executable)

the first one registers a new symbol that is directly tied to the ``torch.Tensor.unfold``, and the second
associates this symbol with ``prims.unfold`` upon execution unless the ``checker`` fails.
Having ``checker=_always_executable`` always greenlights this association, and, hence,
whenever the ``torch`` executor tries to execute ``prims.unfold``, it executes ``torch.Tensor.unfold``.
Note, however, that although the ``checker`` does have access to the symbol's inputs, it is different from the meta-function.
Meta-functions are supposed to only validate inputs and to be executor-agnostic. Checkers, on the other hand, are not
meant to check inputs' validity and they are agnosit to executors. As such, they are useful for checking and enabling
symbols for specific versions of executors like PyTorch, for example.

The mapping of the ``prims.unfold`` symbol to ``torch.Tensor.unfold`` is very simple since the inputs
to ``prims.unfold`` can directly be passed to ``torch.Tensor.unfold`` without any additional pre-preprocessing
(association between ``TensorProxy`` and ``torch.Tensor`` is handled automatically by the ``torch`` executor).
This is not the case with any operation, however, and sometimes the symbol's interface has to undergo
a *transformation* to be compatible with the registered implementation provided by the executor.
For example, the following lines from
`executors/torchex.py <https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/executors/torchex.py>`_

.. code-block:: python
   :lineno-start: 234

   def _full_transform(
       shape: Sequence[int], fill_value: Number, *, device: None | devices.Device, dtype: None | dtypes.dtype
   ) -> TensorProxy:
       torch_device: None | torch.device = to_torch_device(device)
       torch_dtype: None | torch.dtype = to_torch_dtype(dtype)

       return full(shape, fill_value, device=torch_device, dtype=torch_dtype)

.. code-block:: python
   :lineno-start: 421

   _register_implementation(prims.full, checker=_always_executable, execution_transform=_full_transform)

show us how to accomplish that with the ``execution_transform`` argument of ``_register_implementation``
where the *Thunder* meta-data like ``device``, ``dtype`` is converted to the corresponding PyTorch meta-data.

~~~~~~~~~~~~~~~~~~~~~
Testing the Operation
~~~~~~~~~~~~~~~~~~~~~
In the previous section we saw an example of adding a primitive operation.
However, it is not guaranteed that the operation performs as expected.
We need to test it!

Operators are typically tested by adding an OpInfo for them.
See `here <https://github.com/pytorch/pytorch/blob/7a192cc51c7172860efd35413acf9a8b9aafd2c9/torch/testing/_internal/opinfo/core.py#L341>`_
to better understand how OpInfos work.
OpInfo contains metadata describing an operator, a sample input generator,
a sample generator for erroneous inputs that is used for testing handling exceptions/meta function correctness, and test directives.
It’s used to automatically generate a variety of tests, most importantly tests that verify the operator’s behavior is consistent with its reference implementations.

It is important to determine whether you need to add ``test_directives`` in order to skip tests or expect failures of tests.

* Skip (``pytest.mark.skip``): Skips are needed when something is not implemented by an executor or for a device.
* Expected Failures (``pytest.mark.xfail``): Expected failures indicate that an executor has implemented some aspect of an operation but its behavior is incorrect.

An example of OpInfo for ``prims.unfold`` from the `PR #136 <https://github.com/Lightning-AI/lightning-thunder/pull/136>`_
added to `thunder/tests/opinfos.py <https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/tests/opinfos.py>`_

.. code-block:: python
   :lineno-start: 2997
   
   def unfold_sample_generator(op, device, dtype, requires_grad, **kwargs):
       make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

       cases = (
           ((), 0, 1, 3),
           ((), -1, 0, 5),
           ((0,), 0, 0, 1),
           ((8,), 0, 2, 1),
           ((6, 2), 0, 2, 2),
       )

       for shape, dim, size, step in cases:
           yield SampleInput(make(shape), dim, size, step)


    def unfold_error_generator(op, device, dtype=torch.float32, **kwargs):
        make = partial(make_tensor, device=device, dtype=dtype)

        cases = (
            ((), 0, 2, 1, RuntimeError, "Maximum size for tensor at dimension 0 is 1 but size is 2"),
            ((0,), 0, 0, -1, RuntimeError, "Step is -1 but must be > 0"),
            ((8,), 1, 2, 1, IndexError, r"Dimension out of range \(expected to be in range of \[-1, 0\], but got 1\)"),
            ((8,), 0, -5, 1, RuntimeError, "Size is -5 but must be >= 0"),
            ((8,), 0, 10, 1, RuntimeError, "Maximum size for tensor at dimension 0 is 8 but size is 10"),
        )

        for shape, dim, size, step, err_type, err_msg in cases:
            yield SampleInput(make(shape), dim, size, step), err_type, err_msg


    unfold_opinfo = OpInfo(
        clang.unfold,
        sample_input_generator=unfold_sample_generator,
        error_input_generator=unfold_error_generator,
        torch_reference=torch.Tensor.unfold,
    )

    shape_ops.append(unfold_opinfo)

Note how comprehensive ``unfold_sample_generator`` and ``unfold_error_generator`` are.
``unfold_sample_generator`` does not shy away from testing scalar inputs (``shape=()``)
and empty inputs (``shape=(0,)``, i.e. shapes containing zeros).
And ``unfold_error_generator`` tests about every aspect of the underlying meta-function.

To run the tests for a particular operator, use ``pytest``’s ``-k`` option.
This will run tests for *Thunder*’s different executors, supported dtypes, and supported device types.
For example, to run the tests for ``unfold`` the command would be

.. code-block:: bash

  $ pytest thunder/tests/test_ops.py -k unfold

Another example of an OpInfo with specified ``test_directives``

.. code-block:: python
   :lineno-start: 577

   acos_opinfo = OpInfo(
       ltorch.acos,
       domain=(-1, 1),
       sample_input_generator=elementwise_unary_generator,
       torch_reference=_elementwise_unary_torch(torch.acos),
       test_directives=(
           # Torch doesn't support CPU float16 or complex32 acos
           DecorateInfo(
               pytest.mark.xfail,
               "test_core_vs_torch_consistency",
               dtypes=(datatypes.float16, datatypes.complex32),
               devicetypes=(devices.DeviceType.CPU,),
           ),
       ),
   )
   elementwise_unary_ops.append(acos_opinfo)

We strive for *Thunder* to be of the highest quality possible,
so it is always a good idea to be very thorough when it comes to testing.

~~~~~~~~~~~~~~~~~~~
Adding grad support
~~~~~~~~~~~~~~~~~~~
Operations are not differentiable by default, unless they are implemented as compositions
of differentiable operations (related to updating the ``torch`` language. More on that later).
When an operation is a composition of other operations, we say that this operation is *decomposable*.
Primitive operations, by definition, are not decomposable, and, as such, require an explicit
``backward``/``grad``/``VJP`` (for simplicity, we use them interchangeably) rule implemented for them.
These rules, or *grad transforms*, are implemented in
`thunder/core/transforms.py <https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/core/transforms.py>`_.
Note, however, these rules are not exclusively restricted to primitive operations, see
:doc:`Defining custom forward and backward for existing operators <../notebooks/adding_custom_operator_backward>`,
for example, and can be implemented even for *decomposable* operations for performance reasons.

For now, for simplicity, let's assume that a new primitive is being added and we would like to make it differentiable.
Consider the `PR #118 <https://github.com/Lightning-AI/lightning-thunder/pull/118>`_ which adds a backward support for a
primitive operation ``prims.topk`` (added in the `PR #88 <https://github.com/Lightning-AI/lightning-thunder/pull/88>`_)
that is modeled after `torch.topk <https://pytorch.org/docs/stable/generated/torch.topk.html#torch.topk>`_.
The added to
`thunder/core/transforms.py <https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/core/transforms.py>`_
lines

.. code-block:: python
  :lineno-start: 1111

  @torchctx
  def _topk_prim_grad(
      a: TensorProxy, /, k: int, dim: None | int = None, largest: bool = True, sorted: bool = True, *, out=None
  ):
      fwd = prims.topk(a, k, dim, largest, sorted, out=out)
      val, idx = fwd

      val_grad = get_grad(val)

      a_grad = ltorch.zeros_like(a)
      # TODO: replace with scatter once we have it.
      # scatter_add is a prim and it relies on atomic ops.
      a_grad = ltorch.scatter_add(a_grad, dim, idx, val_grad)
      put_grad(a, a_grad)

      return fwd


  register_grad(pids.TOPK, _topk_prim_grad)

define a grad transform for ``prims.topk``.
This operation returns a 2-tuple in forward ``fwd = (val, idx)`` with only the first element
being differentiable. Note that *Thunder* interleaves forward and backward computations in grad transforms.
Take a look at the lines ``val_grad = get_grad(val)``, which extracts the in-flowing backward gradient
for ``val``, and ``put_grad(a, a_grad)`` which sets the backward gradient for the input ``a``.

Do you see that comment about the missing ``scatter``? You could be the one who implements it! :)


~~~~~~~~~~~~~~~~~~~~~~~~~~~
Updating the Torch Language
~~~~~~~~~~~~~~~~~~~~~~~~~~~
`The Torch Language`_ operations are the "highest"-level operations and, as such, are *decomposable*.
If the missing operation can be decomposed into already existing operations, then
`thunder/torch/__init__.py <https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/torch/__init__.py>`_
is where its implementation is to be placed.

For example, consider the `PR #100, <https://github.com/Lightning-AI/lightning-thunder/pull/100>`_ that adds
support for the Hardswish activation function.
The function is implemented in `thunder/torch/__init__.py <https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/torch/__init__.py>`_

.. code-block:: python
    :lineno-start: 1211

    @torchsymbol(torch.nn.functional.hardswish, id="torch.hardswish", is_method=False)
    def hardswish(a: TensorProxy, /, inplace: bool = False) -> TensorLike:
        utils.check(not inplace, lambda: f"hardswish only supports inplace=False", exception_type=NotImplementedError)
        utils.check(
            dtypes.is_float_dtype(a.dtype),
            lambda: f"hardswish only supports floating point dtypes, got {a.dtype}",
            exception_type=ValueError,
        )
        return a * relu6(a + 3) / 6

Note the checks (*Thunder* does not support in-place operations yet) and that ``hardswish`` is a composition
of the ``relu6`` operation (defined in the ``torch`` language) and the language context-specific binary operations
over the objects that ``TensorProxy`` represent. All these basic operations are differentiable
(for the Torch/NVFuser executors), and so is ``hardswish`` implicitly differentiable (for the Torch/NVFuser executors).


=========
Afterword
=========

We hope that you find information provided here useful and we look forward to your contributions!

We also recommend checking out
:doc:`Defining new Thunder operations <../notebooks/adding_custom_operator>` and
:doc:`Defining custom forward and backward for existing operators <../notebooks/adding_custom_operator_backward>`
that cover very similar topics related to extending *Thunder* out of the tree.


