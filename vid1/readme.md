# Introduction to PyTorch

## Origins of PyTorch

PyTorch is an open-source machine learning framework developed by **Meta (formerly Facebook)**. It has its roots in a 2002 scientific computing framework called **Torch**, which was built to support **tensor-based operations on GPUs**. 

Torch gained popularity among deep learning researchers, who started building model implementations on top of it. However, **Torch was based on the Lua programming language**, which wasn’t widely adopted outside of specific communities.

To lower the adoption barrier and cater to the growing popularity of Python in the ML ecosystem, Meta released **PyTorch**, a **Python-based deep learning framework** that builds upon the ideas of Torch while offering a more flexible and Pythonic interface.

---

## Key Concepts

### Dynamic Computation Graphs

Unlike other frameworks at the time (e.g., TensorFlow 1.x), which relied on **static computation graphs** (i.e., graphs defined once and compiled before running), PyTorch introduced **Dynamic Computation Graphs** (also called *define-by-run*). 

This means:
- The graph is built on-the-fly as operations are executed.
- It is intuitive and debuggable using standard Python debugging tools (like `print()` and `pdb`).
- Especially powerful for use-cases like **variable-length sequences**, **recurrent networks**, and **research experimentation**.

> A computation graph is a visual and functional representation of the mathematical operations your model performs, especially useful in automatic differentiation and optimization.

---

## Core Features

- **Tensor Computations** — Multi-dimensional arrays (like NumPy), with strong GPU acceleration support.
- **GPU Acceleration** — Easily move tensors and models between CPU and GPU.
- **Dynamic Computation Graph** — Define models dynamically as code runs.
- **Automatic Differentiation** — Built-in backpropagation with `torch.autograd`.
- **Distributed Training** — Supports parallel training across multiple devices and nodes.
- **Interoperability** — Easily integrates with libraries like NumPy, SciPy, OpenCV, and ONNX.

---

## Core Modules

### 1. **`torch`**
- Core namespace that provides tensor creation, mathematical operations, and many utilities.

### 2. **`torch.autograd`**
- Enables **automatic differentiation** — a key part of training neural networks.

### 3. **`torch.nn`**
- Provides **high-level abstractions** to build and train neural networks, including layers, loss functions, and model containers like `Sequential`.

### 4. **`torch.optim`**
- Contains various optimization algorithms (SGD, Adam, RMSprop, etc.) used in model training.

### 5. **`torch.utils.data`**
- Offers tools for:
  - Custom **datasets** (via `Dataset` class)
  - Efficient **batch loading** and preprocessing (via `DataLoader`)

### 6. **`torch.jit`**
- Part of the **TorchScript** framework, allowing you to:
  - Serialize and optimize models
  - Run them outside of Python (e.g., C++ runtime)

### 7. **`torch.distributed`**
- Enables training across **multiple GPUs and machines** — supports various backends like NCCL, Gloo, MPI.

### 8. **`torch.cuda`**
- GPU support:
  - Move tensors/models to GPU with `.to('cuda')` or `.cuda()`
  - Check availability with `torch.cuda.is_available()`

### 9. **`torch.backends`**
- Configuration and low-level control for things like:
  - cuDNN behavior
  - Deterministic training
  - Debugging numerical issues

### 10. **`torch.multiprocessing`**
- Enables multiprocessing with GPU support — helpful for large-scale data processing and model parallelism.

### 11. **`torch.quantization`**
- Tools for **model quantization**, which:
  - Reduces model size
  - Speeds up inference
  - Lowers power consumption — useful for deploying models on edge devices

### 12. **`torch.onnx`**
- Export models to the **Open Neural Network Exchange (ONNX)** format.
- ONNX allows interoperability with other tools like TensorRT, ONNX Runtime, etc.

