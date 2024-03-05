Train a MLP on MNIST
####################

Here's a complete program that trains a torchvision MLP on MNIST::

  pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu121

Here's the code::

  import os
  import torch
  import torchvision
  import torchvision.transforms as transforms
  import thunder

  # Creates train and test datasets
  device = 'cuda'

  device_transform = transforms.Lambda(lambda t: t.to(device))
  flatten_transform = transforms.Lambda(lambda t: t.flatten())
  my_transform = transforms.Compose((transforms.ToTensor(), device_transform, flatten_transform))

  train_dataset = torchvision.datasets.MNIST("/tmp/mnist/train", train=True, download=True, transform=my_transform)
  test_dataset = torchvision.datasets.MNIST("/tmp/mnist/test", train=False, download=True, transform=my_transform)

  # Creates Samplers
  train_sampler = torch.utils.data.RandomSampler(train_dataset)
  test_sampler = torch.utils.data.RandomSampler(test_dataset)

  # Creates DataLoaders
  batch_size = 8
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

  # Evaluates the model
  def eval_model(model, test_loader):
      num_correct = 0
      total_guesses = 0
      for data, targets in iter(test_loader):
          targets = targets.cuda()
          # Acquires the model's best guesses at each class
          results = model(data)
          best_guesses = torch.argmax(results, 1)
          # Updates number of correct and total guesses
          num_correct += torch.eq(targets, best_guesses).sum().item()
          total_guesses += batch_size
      # Prints output
      print("Correctly guessed ", (num_correct/total_guesses) * 100, "% of the dataset")

  # Trains the model
  def train_model(model, train_loader, *, num_epochs: int = 1):
      loss_fn = torch.nn.CrossEntropyLoss().to(device)
      optimizer = torch.optim.Adam(model.parameters())
      for epoch in range(num_epochs):
          for data, targets in iter(train_loader):
              targets = targets.cuda()
              # Acquires the model's best guesses at each class
              results = model(data)
              # Computes loss
              loss = loss_fn(results, targets)
              # Updates model
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

  # Constructs the model
  model = torchvision.ops.MLP(in_channels=784, hidden_channels=[784, 784, 784, 28, 10], bias=True, dropout=.1).to(device)

  # Performs an initial evaluation
  model.eval().requires_grad_(False)
  jitted_eval_model = thunder.jit(model)
  eval_model(jitted_eval_model, test_loader)

  # Trains the model
  model.train().requires_grad_(True)
  jitted_train_model = thunder.jit(model)
  train_model(jitted_train_model, train_loader)

  model.eval().requires_grad_(False)

  # Performs a final evaluation
  eval_model(jitted_eval_model, test_loader)

  # Evaluates the original, unjitted model
  # The unjitted and jitted model share parameters, so it's
  # also updated
  eval_model(model, test_loader)

  # Acquires and prints thunder's "traces", which show what thunder executed
  # The training model has both "forward" and "backward" traces, corresponding
  # to its forward and backward computations.
  # The evaluation model has only one set of traces.
  fwd_traces, bwd_traces = thunder.last_traces(jitted_train_model)
  eval_traces = thunder.last_traces(jitted_eval_model)

  print("This is the trace that thunder executed for training's forward computation:")
  print(fwd_traces[-1])

  print("This is the trace that thunder executed for training's backward computation:")
  print(bwd_traces[-1])

  print("This is the trace that thunder executed for eval's computation:")
  print(eval_traces[-1])

Let's look at a few parts of this program more closely.

First, up until the call to ``thunder.jit()`` the program is just Python, PyTorch and torchvision. ``thunder.jit()`` accepts a PyTorch module (or function) and returns a Thunder-optimized module that has the same signature, parameters and buffers.

After compilation the program is, again, just Python and PyTorch, until the very end. Behind the scenes, when a Thunder module is called it produces a “trace” representing the sequence of tensor operations to perform. This trace is then transformed and optimized, and the sequence of these traces for the last inputs can be acquired by calling ``thunder.last_traces()`` on the module (the traced program changes when different input data types, devices, or other properties are used). When the module is used for training, ``thunder.last_traces()`` will return both the sequence of “forward” traces and the sequence of “backward” traces, and when it's just used for evaluation it will just return one sequence of traces. In this case we're printing the last traces in the sequence, which print as Python programs, and these Python programs are what gets executed by Thunder.

Let's take a look at the execution trace for the training module's forward::

  @torch.no_grad()
  @no_autocast()
  def augmented_forward_fn(t0, t4, t5, t21, t22, t38, t39, t55, t56, t72, t73):
    # t0
    # t4
    # t5
    # t21
    # t22
    # t38
    # t39
    # t55
    # t56
    # t72
    # t73
    t1 = torch.nn.functional.linear(t0, t4, t5)  # t1
      # t1 = ltorch.linear(t0, t4, t5)  # t1
        # t1 = prims.linear(t0, t4, t5)  # t1
    [t10, t2, t7] = nvFusion0(t1)
      # t2 = prims.gt(t1, 0.0)  # t2
      # t3 = prims.where(t2, t1, 0.0)  # t3
      # t6 = prims.uniform((8, 784), 0.0, 1.0, device=devices.Device("cuda:0"), dtype=dtypes.float32)  # t6
      # t7 = prims.lt(t6, 0.9)  # t7
      # t8 = prims.convert_element_type(t7, dtypes.float32)  # t8
      # t9 = prims.mul(t3, t8)  # t9
      # t10 = prims.mul(t9, 1.1111111111111112)  # t10
    del t1
    t11 = torch.nn.functional.linear(t10, t21, t22)  # t11
      # t11 = ltorch.linear(t10, t21, t22)  # t11
        # t11 = prims.linear(t10, t21, t22)  # t11
    [t12, t15, t18] = nvFusion1(t11)
      # t12 = prims.gt(t11, 0.0)  # t12
      # t13 = prims.where(t12, t11, 0.0)  # t13
      # t14 = prims.uniform((8, 784), 0.0, 1.0, device=devices.Device("cuda:0"), dtype=dtypes.float32)  # t14
      # t15 = prims.lt(t14, 0.9)  # t15
      # t16 = prims.convert_element_type(t15, dtypes.float32)  # t16
      # t17 = prims.mul(t13, t16)  # t17
      # t18 = prims.mul(t17, 1.1111111111111112)  # t18
    del t11
    t19 = torch.nn.functional.linear(t18, t38, t39)  # t19
      # t19 = ltorch.linear(t18, t38, t39)  # t19
        # t19 = prims.linear(t18, t38, t39)  # t19
    [t20, t25, t28] = nvFusion2(t19)
      # t20 = prims.gt(t19, 0.0)  # t20
      # t23 = prims.where(t20, t19, 0.0)  # t23
      # t24 = prims.uniform((8, 784), 0.0, 1.0, device=devices.Device("cuda:0"), dtype=dtypes.float32)  # t24
      # t25 = prims.lt(t24, 0.9)  # t25
      # t26 = prims.convert_element_type(t25, dtypes.float32)  # t26
      # t27 = prims.mul(t23, t26)  # t27
      # t28 = prims.mul(t27, 1.1111111111111112)  # t28
    del t19
    t29 = torch.nn.functional.linear(t28, t55, t56)  # t29
      # t29 = ltorch.linear(t28, t55, t56)  # t29
        # t29 = prims.linear(t28, t55, t56)  # t29
    [t30, t33, t36] = nvFusion3(t29)
      # t30 = prims.gt(t29, 0.0)  # t30
      # t31 = prims.where(t30, t29, 0.0)  # t31
      # t32 = prims.uniform((8, 28), 0.0, 1.0, device=devices.Device("cuda:0"), dtype=dtypes.float32)  # t32
      # t33 = prims.lt(t32, 0.9)  # t33
      # t34 = prims.convert_element_type(t33, dtypes.float32)  # t34
      # t35 = prims.mul(t31, t34)  # t35
      # t36 = prims.mul(t35, 1.1111111111111112)  # t36
    del t29
    t37 = torch.nn.functional.linear(t36, t72, t73)  # t37
      # t37 = ltorch.linear(t36, t72, t73)  # t37
        # t37 = prims.linear(t36, t72, t73)  # t37
    [t41, t44] = nvFusion4(t37)
      # t40 = prims.uniform((8, 10), 0.0, 1.0, device=devices.Device("cuda:0"), dtype=dtypes.float32)  # t40
      # t41 = prims.lt(t40, 0.9)  # t41
      # t42 = prims.convert_element_type(t41, dtypes.float32)  # t42
      # t43 = prims.mul(t37, t42)  # t43
      # t44 = prims.mul(t43, 1.1111111111111112)  # t44
    del t37
    return {'output': (t44, ()), 'flat_args': [t0, t4, t5, t21, t22, t38, t39, t55, t56, t72, t73], 'flat_output': (t44,)}, ((t0, t10, t12, t15, t18, t2, t20, t21, t25, t28, t30, t33, t36, t38, t41, t55, t7, t72), (1.1111111111111112, 1.1111111111111112, 1.1111111111111112, 1.1111111111111112, 1.1111111111111112))

There's a lot going on here, and if you'd like to get into the details then keep reading! But we can see that the trace is a functional Python function, and Thunder has produced several groups of primitives that are sent to nvFuser. Instead of leaving these primitives directly in the module, nvFuser has produced several optimized kernels (fusions) and inserted them into the program (``nvFusion0``, ``nvFusion1``, ...). Under each fusion (in comments) are the “primitive” operations that describe precisely what each group does, although how each fusion is executed is entirely up to nvFuser.
