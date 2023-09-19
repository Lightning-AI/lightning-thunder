Train a MLP on MNIST
####################

Here's a complete program that trains a torchvision MLP on MNIST::

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
  train_dataset = torchvision.datasets.MNIST(
    os.path.join("/tmp/mnist/train"),
    train=True,
    download=True,
    transform=my_transform)
  test_dataset = torchvision.datasets.MNIST(
    os.path.join("/tmp/mnist/test"),
    train=False,
    download=True,
    transform=my_transform)

  # Creates Samplers
  train_sampler = torch.utils.data.RandomSampler(train_dataset)
  test_sampler = torch.utils.data.RandomSampler(test_dataset)

  # Creates DataLoaders
  batch_size = 8
  train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler)
  test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    sampler=test_sampler)

  # Evaluates the model
  def eval_model(net, test_loader):
      num_correct = 0
      total_guesses = 0
      for data, targets in iter(test_loader):
          targets = targets.cuda()
          # Acquires the model's best guesses at each class
          results = net(data)
          best_guesses = torch.argmax(results, 1)
          # Updates number of correct and total guesses
          num_correct += torch.eq(targets, best_guesses).sum().item()
          total_guesses += batch_size
      # Prints output
      print("Correctly guessed ", (num_correct/total_guesses) * 100, "% of the dataset")

  # Trains the model
  def train_model(net, train_loader, *, num_epochs: int = 1):
      loss_fn = torch.nn.CrossEntropyLoss().to(device)
      optimizer = torch.optim.Adam(net.parameters())
      for epoch in range(num_epochs):
          for data, targets in iter(train_loader):
              targets = targets.cuda()
              # Acquires the model's best guesses at each class
              results = net(data)
              # Computes loss
              loss = loss_fn(results, targets)
              # Updates model
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

  # Constructs the model
  net = torchvision.ops.MLP(in_channels=784, hidden_channels=[784, 784, 784, 28, 10], bias=True, dropout=.1).to(device)

  # Performs an initial evaluation
  net.eval().requires_grad_(False)
  compiled_eval_net = thunder.compile(net)
  eval_model(compiled_eval_net, test_loader)

  # Trains the model
  net.train().requires_grad_(True)
  compiled_train_net = thunder.compile(net)
  train_model(compiled_train_net, train_loader)
  net.eval().requires_grad_(False)

  # Performs a final evaluation
  eval_model(compiled_eval_net, test_loader)

  # Evaluates the original, uncompiled model
  # The uncompiled and compiled model share parameters, so it's
  # also updated
  eval_model(net, test_loader)

  # Acquires and prints thunder's "traces", which show what thunder executed
  # The training model has both "forward" and "backward" traces, corresponding
  # to its forward and backward computations.
  # The evaluation model has only one set of traces.
  fwd_traces, bwd_traces = thunder.last_traces(compiled_train_net)
  eval_traces = thunder.last_traces(compiled_eval_net)
  print("This is the trace that thunder executed for training's forward computation:")
  print(fwd_traces[-1])
  print("This is the trace that thunder executed for training's backward computation:")
  print(bwd_traces[-1])
  print("This is the trace that thunder executed for eval's computation:")
  print(eval_traces[-1])

Let's look at a few parts of this program more closely.

First, up until the call to ``thunder.compile()`` the program is just Python, PyTorch and torchvision. ``thunder.compile()`` accepts a PyTorch module (or function) and returns a “Thunder Optimized Module (TOM)”. The TOM's input signature and outputs will match exactly the PyTorch module or function passed to compile. The TOM is simply an optimized version and is expected to be used as a direct replacement. Furthermore, the PyTorch module and the TOM share their parameters and buffers, as we'll see in a moment.

There's a lot that goes into running a TOM, and we'll take a peek behind the scenes in a moment. What's important to know, however, is that we have to ``thunder.compile()`` our model twice if we want to train and evaluate it. This is because the metadata of a module's parameters is assumed to be constant after the TOM is first run, so it will ignore calls to ``requires_grad_()``. This assumption is a performance optimization, and it'll be easier to work directly with TOMs in the future.

After compilation the program is, again, just Python and PyTorch, until the very end. Behind the scenes, when a TOM is called it produces a “trace” representing the sequence of tensor operations to perform. This trace is then transformed and optimized, and the sequence of these traces for the last inputs can be acquired by calling ``thunder.last_traces()`` on the TOM (the traced program changes when different input data types, devices, or other properties are used). When the TOM is used for training, ``thunder.last_traces()`` will return both the sequence of “forward” traces and the sequence of “backward” traces, and when it's just used for evaluation it will just return one sequence of traces. In this case we're printing the last traces in the sequence, which print as Python programs, and these Python programs are what gets executed by *thunder*.

Let's take a look at the execution trace for the training TOM's forward::

  @torch.no_grad()
  def augmented_forward_fn(_0_weight, _0_bias, _3_weight, _3_bias, _6_weight, _6_bias, _9_weight, _9_bias, _12_weight, _12_bias, input):
    # _0_weight: "cuda:0 f32[784, 784]"
    # _0_bias: "cuda:0 f32[784]"
    # _3_weight: "cuda:0 f32[784, 784]"
    # _3_bias: "cuda:0 f32[784]"
    # _6_weight: "cuda:0 f32[784, 784]"
    # _6_bias: "cuda:0 f32[784]"
    # _9_weight: "cuda:0 f32[28, 784]"
    # _9_bias: "cuda:0 f32[28]"
    # _12_weight: "cuda:0 f32[10, 28]"
    # _12_bias: "cuda:0 f32[10]"
    # input: "cuda:0 f32[8, 784]"
    t0 = torch.nn.functional.linear(input, _0_weight, _0_bias)  # t0: "cuda:0 f32[8, 784]"
    (t1, t4, t7) = nvFusion0(t0)
      # t1 = prims.gt(t0, 0.0)  # t1: "cuda:0 b8[8, 784]"
      # t2 = prims.where(t1, t0, 0.0)  # t2: "cuda:0 f32[8, 784]"
      # t3 = prims.uniform((8, 784), 0.0, 1.0, device=devices.Device("cuda:0"), dtype=dtypes.float32)  # t3: "cuda:0 f32[8, 784]"
      # t4 = prims.lt(t3, 0.9)  # t4: "cuda:0 b8[8, 784]"
      # t5 = prims.convert_element_type(t4, dtypes.float32)  # t5: "cuda:0 f32[8, 784]"
      # t6 = prims.mul(t2, t5)  # t6: "cuda:0 f32[8, 784]"
      # t7 = prims.mul(t6, 1.1111111111111112)  # t7: "cuda:0 f32[8, 784]"
    del [t0]
    t8 = torch.nn.functional.linear(t7, _3_weight, _3_bias)  # t8: "cuda:0 f32[8, 784]"
    (t12, t15, t9) = nvFusion1(t8)
      # t9 = prims.gt(t8, 0.0)  # t9: "cuda:0 b8[8, 784]"
      # t10 = prims.where(t9, t8, 0.0)  # t10: "cuda:0 f32[8, 784]"
      # t11 = prims.uniform((8, 784), 0.0, 1.0, device=devices.Device("cuda:0"), dtype=dtypes.float32)  # t11: "cuda:0 f32[8, 784]"
      # t12 = prims.lt(t11, 0.9)  # t12: "cuda:0 b8[8, 784]"
      # t13 = prims.convert_element_type(t12, dtypes.float32)  # t13: "cuda:0 f32[8, 784]"
      # t14 = prims.mul(t10, t13)  # t14: "cuda:0 f32[8, 784]"
      # t15 = prims.mul(t14, 1.1111111111111112)  # t15: "cuda:0 f32[8, 784]"
    del [t8]
    t16 = torch.nn.functional.linear(t15, _6_weight, _6_bias)  # t16: "cuda:0 f32[8, 784]"
    (t17, t20, t23) = nvFusion2(t16)
      # t17 = prims.gt(t16, 0.0)  # t17: "cuda:0 b8[8, 784]"
      # t18 = prims.where(t17, t16, 0.0)  # t18: "cuda:0 f32[8, 784]"
      # t19 = prims.uniform((8, 784), 0.0, 1.0, device=devices.Device("cuda:0"), dtype=dtypes.float32)  # t19: "cuda:0 f32[8, 784]"
      # t20 = prims.lt(t19, 0.9)  # t20: "cuda:0 b8[8, 784]"
      # t21 = prims.convert_element_type(t20, dtypes.float32)  # t21: "cuda:0 f32[8, 784]"
      # t22 = prims.mul(t18, t21)  # t22: "cuda:0 f32[8, 784]"
      # t23 = prims.mul(t22, 1.1111111111111112)  # t23: "cuda:0 f32[8, 784]"
    del [t16]
    t24 = torch.nn.functional.linear(t23, _9_weight, _9_bias)  # t24: "cuda:0 f32[8, 28]"
    (t25, t28, t31) = nvFusion3(t24)
      # t25 = prims.gt(t24, 0.0)  # t25: "cuda:0 b8[8, 28]"
      # t26 = prims.where(t25, t24, 0.0)  # t26: "cuda:0 f32[8, 28]"
      # t27 = prims.uniform((8, 28), 0.0, 1.0, device=devices.Device("cuda:0"), dtype=dtypes.float32)  # t27: "cuda:0 f32[8, 28]"
      # t28 = prims.lt(t27, 0.9)  # t28: "cuda:0 b8[8, 28]"
      # t29 = prims.convert_element_type(t28, dtypes.float32)  # t29: "cuda:0 f32[8, 28]"
      # t30 = prims.mul(t26, t29)  # t30: "cuda:0 f32[8, 28]"
      # t31 = prims.mul(t30, 1.1111111111111112)  # t31: "cuda:0 f32[8, 28]"
    del [t24]
    t32 = torch.nn.functional.linear(t31, _12_weight, _12_bias)  # t32: "cuda:0 f32[8, 10]"
    (t34, t37) = nvFusion4(t32)
      # t33 = prims.uniform((8, 10), 0.0, 1.0, device=devices.Device("cuda:0"), dtype=dtypes.float32)  # t33: "cuda:0 f32[8, 10]"
      # t34 = prims.lt(t33, 0.9)  # t34: "cuda:0 b8[8, 10]"
      # t35 = prims.convert_element_type(t34, dtypes.float32)  # t35: "cuda:0 f32[8, 10]"
      # t36 = prims.mul(t32, t35)  # t36: "cuda:0 f32[8, 10]"
      # t37 = prims.mul(t36, 1.1111111111111112)  # t37: "cuda:0 f32[8, 10]"
    del [t32]
    return {'output': t37, 'flat_args': [_0_weight, _0_bias, _3_weight, _3_bias, _6_weight, _6_bias, _9_weight, _9_bias, _12_weight, _12_bias, input], 'flat_output': (t37,)}, ((_12_weight, _3_weight, _6_weight, _9_weight, input, t1, t12, t15, t17, t20, t23, t25, t28, t31, t34, t4, t7, t9), (1.1111111111111112, 1.1111111111111112, 1.1111111111111112, 1.1111111111111112, 1.1111111111111112))

There's a lot going on here, and if you'd like to get into the details then keep reading! But we can see that the trace is a functional Python function, and *thunder* has produced several groups of primitives that are sent to nvFuser. Instead of leaving these primitives directly in the TOM, nvFuser has produced several optimized kernels (fusions) and inserted them into the program (``nvFusion0``, ``nvFusion1``, ...). Under each fusion (in comments) are the “primitive” operations that describe precisely what each group does, although how each fusion is executed is entirely up to nvFuser.
