Single-GPU benchmarking in Thunder
##################################

After reading this section you'll be able to understand what benchmarks are available in Thunder, how to run them and how to create one yourself.

Introduction
============
In Thunder there are a two of ways of benchmarking the compiler:

- One is by running a synthetic end-to-end training of a model from `LitGPT <https://github.com/Lightning-AI/litgpt/>`__ ; and
- the second is by microbenchmarking specific snippets of code.

Before starting, you need to install Thunder and the devel packages with::

  pip install -r requirements/devel.txt
  pip install -e .

LitGPT benchmarks
=================

The esiest way to run a single benchmark in Thunder is by running an instance of the LitGPT end-to-end training script that can be found in ``thunder/benchmarks/benchmark_litgpt.py``.

To run a benchmark all we need is the following command::

  python thunder/benchmarks/benchmark_litgpt.py --model_name <model name> --compile "thunder"


All the command line options can be queried by passing ``--help`` as argument or `can be seen here <https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/benchmarks/benchmark_litgpt.py#L103-L120>`_. However, the most important options for single GPU benchmarks are:

- ``--compile``: specifies the compile mode for the run
- ``--model_name``: LitGPT model name, a list of these can be found `here <https://github.com/Lightning-AI/litgpt/?tab=readme-ov-file#choose-from-20-llms>`_
- ``--n_layers``: specifies the number of layers in the model and can be useful to test a reduced version of the model in the case the original version does not fit into memory.
- ``--nsys_enabled``: inserts markers to profile the run with `NVIDIA Nsight System <https://developer.nvidia.com/nsight-systems/get-started>`_

The output from this end-to-end benchmark will look something like this::

  Time to instantiate model: 0.12 seconds.
  iter 0: loss 10.5000, iter time: 73490.96ms, t: 4096
  ...
  iter 44: loss 4.6250, iter time: 385.25ms, t: 4096
  Model name: Llama-2-7b-hf
  Seq Length: 4096
  Micro BS: 1
  Global BS: 1
  Number of Layers: 32
  Number of parameters: 6.74B
  Distributed Mode: none
  Compiler: thunder
  Low Precision Mode: none
  Average iter time: 383.11 ms
  Memory used: 64.22 GB
  Tokens/s: 10690.01
  Tokens/s/GPU: 10690.01
  TFLOP/s: 492.65

.. note:: Beware the memory footprint of certain models! In this example, running ``Llama-2-7b-hf`` on H100 with default Thunder compile option requires upwards of ~65GB of memory. Pro tip: you can always play with the ``--n_layers`` option to run reduced versions of the model that can fit in memory.

Compile options
---------------

With the ``--compile`` option, you can test:

- torch.compile by specifying ``inductor``
- torch eager mode by specifying ``eager``, or
- Thunder by specifying ``thunder``

To customize Thunder executors, in addition to nvFuser, you can append the any combination of the following to the string with an underscore:

- ``inductor_cat``
- ``cudnn``
- ``transformerengine``

As an example, if you want to use cudnn as executor your terminal will looks something like::

  python thunder/benchmarks/benchmark_litgpt.py --model_name Llama-2-7b-hf  --compile thunder_cudnn


and if you are testing ``torch.compile`` then it will look something like this::

  python thunder/benchmarks/benchmark_litgpt.py --model_name Llama-2-7b-hf  --compile inductor

pytest benchmarks
=================

If instead of running an e2e training benchamrk you want to be more specific, Thunder has you covered with the pytest based benchmarks (more specifically `pytest-benchmark <https://pytest-benchmark.readthedocs.io/en/latest/>`__).
These benchamrks are defined in two parts, the implementation is in ``thunder/benchmarks/__init__.py`` and the hook for pytest is in ``thunder/benchmarks/targets.py``.
In the next section you'll see more of the details, but for now let's start by listing all the available benchmarks with::

  pytest thunder/benchmarks/targets.py --collect-only

To run all the available benchmarks, it's as simple as calling::

  pytest thunder/benchmarks/targets.py

However, more realistically you'd want to filter and run just specific benchmarks. To do so, you can use `the filter syntax <https://docs.pytest.org/en/stable/how-to/usage.html#specifying-which-tests-to-run>`_ along with the `-k` option::

  pytest thunder/benchmarks/targets.py -k 'nanogpt_gpt2 and not torch.compile and not xl and not inference' --benchmark-group-by='param:compute_type'

This example will select the benchmarks run them and print the results grouped the results by compute type(forward and backward in this case) thanks to the ``--benchmark-group-by`` flag.
The output will look something like this(it's pretty wide so it looks a bit wierd on narrow windows)::

  ------------------------------------------------------------------- benchmark 'compute_type=ComputeType.TRAINING_BACKWARD': 2 tests ---------------------------------------------------------------
  Name (time in ms)                           Min                Max               Mean            StdDev             Median               IQR            Outliers      OPS        Rounds  Iterations
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  test_nanogpt_gpt2[backward-torch]       11.1503 (1.0)      11.7122 (1.0)      11.2785 (1.0)      0.0973 (1.65)     11.2674 (1.0)      0.1069 (1.12)         16;4  88.6641 (1.0)      93           1
  test_nanogpt_gpt2[backward-thunder]     11.4634 (1.03)     11.7805 (1.01)     11.6194 (1.03)     0.0590 (1.0)      11.6087 (1.03)     0.0952 (1.0)          28;0  86.0632 (0.97)     91           1
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ------------------------------------------------------------------ benchmark 'compute_type=ComputeType.TRAINING_FORWARD': 2 tests -----------------------------------------------------------------
  Name (time in ms)                         Min               Max              Mean            StdDev            Median               IQR            Outliers       OPS            Rounds  Iterations
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  test_nanogpt_gpt2[forward-torch]       5.0307 (1.0)      5.5468 (1.0)      5.1072 (1.0)      0.0901 (1.0)      5.0885 (1.0)      0.0402 (1.0)         11;15  195.8038 (1.0)         228           1
  test_nanogpt_gpt2[forward-thunder]     7.5619 (1.50)     8.0979 (1.46)     7.6878 (1.51)     0.1358 (1.51)     7.6421 (1.50)     0.0602 (1.50)        15;15  130.0763 (0.66)        133           1
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
  ================================================================ 4 passed, 598 deselected in 113.92s (0:01:53) ====================================================================================

Comparing pytest runs
---------------------

Another tool at your disposal is the comparison offered by ``pytest-benchmark``::

  pytest thunder/benchmarks/targets.py --benchmark-autosave -k "thunder]"
  [... your changes ...]
  pytest thunder/benchmarks/targets.py --benchmark-autosave -k "thunder]"
  pytest-benchmark compare 0001 0002 --group-by='name'

By using ``--autosave`` pytest will save the results so that you can read or compare them later.

Writing your own benchmark
==========================

Now that you've seen how the benchmarks work, it's time to add your own benchmark to Thunder by:

1. Creating a class that is a subclass of ``thunder.benchmark.Benchmark`` and define it's methods;
2. Declaring a function with name starting with ``test_`` that uses the class created in the previous step; and
3. Parametrizing the function with all the options needed.

Let's take a deeper dive for each point.

Creating a benchmarking class
-----------------------------

As stated before, you need to create a class that inherits from ``thunder.benchmark.Benchmark`` as following::

  from thunder.benchmarks import Benchmark, BenchmarkArg

  class FooBenchmark(Benchmark):
      @classmethod
      @property
      def name(cls) -> str:
          return "foo_bench"

      @classmethod
      @property
      def description(cls) -> str:
          return "Benchmark for foo function"

.. note:: The ``name`` should be short, distinct, and a valid filename like "nanogpt" or "llamba-block" and
    the ``description`` should be a short sentence describing the benchmark like "NanoGPT's LayerNorm module forward".

The next step is to declare a list of accepted arguments from this benchmark as a property of the class and a class method that returns those arguments::

      _args = (
          BenchmarkArg(name="device", description="A string representing the device. Default is 'cuda'."),
          BenchmarkArg(name="dtype", description="The dtype of the tensors. Default is thunder.float32."),
      )

      @classmethod
      @property
      def args(cls) -> tuple[BenchmarkArg, ...]:
          return cls._args

Now that the arguments are setup, the ``__init__()`` method must be implemented::

      def __init__(self, device="cuda", dtype=thunder.float32):
          super().__init__(self)
          self.device: str = device
          self.dtype: dtypes.dtype = dtype

.. note:: ``__init__()`` should call ``super()`` and it can accept additional optional parameters, like parameters with default values or kwargs other than the ``BenchmarkArg``, but these parameters must be after the benchmark arg parameters.

Next, you'll want to create the data for your benchamrk. To do so, you must implement a ``make_batch()`` method that prepares a valid input for the benchamrk, possibly modified by the initialization arguments::

      def make_batch(self) -> tuple[list, dict]:
          make = partial(make_tensor, device=self.device, dtype=self.dtype)
          return (make(10, 10),), {}

Now comes the best part, the ``fn()`` method, which should return the callable that will be benchmarked. The return callable should accept the output of ``make_batch()`` ::

      def fn(self) -> Callable:
          def foo(a):
              return a + a

          return foo

If your benchmark doesn't need any futher steps you'd be done here howerver, consider the case where you want to benchmark a model, then you ``fn()`` method would look something like::

      def fn(self) -> Callable:
          class FooNetwork(torch.nn.Module):
              def __init__(self):
                  super().__init__()
                  self.layer = torch.nn.Linear(10, 10)

              def forward(self, x):
                  return self.layer(x)

          foo = FooNetwork().to(device=self.device, dtype=self.dtype).requires_grad_()
          return foo

Now this is just half of the test, what about the backward pass? In this case, you'll need to implement a ``postprocess_for_backward()`` method to take care of that::

      def postprocess_for_backward(self, out: torch.Tensor) -> torch.Tensor | None:
          # Check if backward it's needed at all
          if not self.requires_grad:
              return

          targets = make_tensor_like(out)  # fake targets
          loss = torch.nn.functional.mse_loss(out, targets)
          return loss

.. note:: This method will be given the output of fn(), and if it returns a torch.Tensor t that requires grad then the benchmark will call t.backward(torch.randn_like(t)).
  By default, postprocess_for_backward() returns the output of fn(), or the first element of the output of fn() if fn() returns a Sequence.


Declaring a test function and its parametrization
-------------------------------------------------

Now that your benchmarking class is ready you have nowhere to call it. To address this issue, let's write a ``test_`` prefixed function in ``thunder/benchmarks/targets.py`` that will use the newly created ``FooBenchmark`` class::

  def test_foo(benchmark):
      bench: Benchmark = FooBenchmark(device="cuda", dtype=thunder.bfloat16)

      args, kwargs = bench.make_batch()
      benchmark(bench.fn(), *args, **kwargs)

Great! You are ready to benchmark ``foo()``! But what if you want to test it with different Thunder executors? Here comes parametrization to help. To parametrize the function all it's needed it's the use of the ``@pytest.mark.parametrize`` decorator as following::

  @pytest.mark.parametrize(
      "executor",
      (
          torch_executor,
          torch_compile_executor,
          thunder_executor,
      ),
      ids=("torch", "torch.compile", "thunder"),
  )
  def test_foo(benchmark, executor):
      bench: Benchmark = FooBenchmark(device="cuda", dtype=thunder.bfloat16)

      args, kwargs = bench.make_batch()
      fn = executor(bench.fn())

      benchmark(fn, *args, **kwargs)

Here you go, now you are ready to start benchmarking! For more information about the parametrization syntax you can `get a look here <https://docs.pytest.org/en/8.2.x/how-to/parametrize.html>`_.

Benchmarking forward and backward separately
--------------------------------------------

As seen earlier, it's possible to write benchmarks for models and not just standalone functions. What if you want to benchmark forward and backward pass separately? It's possible by tweaking the ``test_`` function you just declared in ``thunder/benchmarks/targets.py`` like so::

  #[...previous parametrization omitted here...]
  @parametrize_compute_type
  def test_foo(benchamrk, executor, compute_type: ComputeType):
      bench: Benchmark = FooBenchmark(device="cuda", dtype=thunder.bfloat16)

      args, kwargs = bench.make_batch()
      fn = executor(bench.fn())

      benchmark_for_compute_type(compute_type, benchamrk, fn, *args, **kwargs)

And that's as simple as that! Just add the decorator ``@parametrize_compute_type`` after your parametrization, add the ``compute_type`` argument, and use ``benchmark_for_compute_type`` to call the benchmark function.
