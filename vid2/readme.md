# Tensors in Deep Learning

Tensors are the **core data structure** in deep learning — a flexible, efficient way to represent and manipulate multidimensional data. This README gives you a comprehensive overview of tensors, how they work, and why they're indispensable in the world of deep learning.

---

## What is a Tensor?

A **tensor** is a **multidimensional array**, a generalized form of scalars, vectors, and matrices. It is a data structure that can be used to store data of varying dimensions, and is optimized for efficient mathematical operations, especially when working with GPUs and TPUs.

> In deep learning frameworks like PyTorch and TensorFlow, everything — data, weights, gradients — is represented as tensors.

---

## Tensor Dimensions

| Tensor Type   | Description                                                  | Shape Example                |
|---------------|--------------------------------------------------------------|------------------------------|
| **0D (Scalar)**   | A single number                                              | `()`                         |
| **1D (Vector)**   | A list of numbers                                             | `(4,)`                       |
| **2D (Matrix)**   | Rows and columns                                              | `(3, 4)`                     |
| **3D Tensor**     | Stacks of matrices — common in images                         | `(256, 256, 3)`              |
| **4D Tensor**     | Batches of 3D tensors — used for image training               | `(32, 128, 128, 3)`          |
| **5D Tensor**     | Adds a time dimension — used for video or sequence data      | `(10, 16, 64, 64, 3)`        |

### Examples:

- RGB Image (256x256): `shape = (256, 256, 3)`
- Batch of 32 RGB Images (128x128): `shape = (32, 128, 128, 3)`
- Batch of 10 Video Clips (16 frames, 64x64, 3 channels): `shape = (10, 16, 64, 64, 3)`

---

## Why Are Tensors Useful?

1. **Mathematical Operations**  
   Tensors support vectorized operations (e.g., dot products, matrix multiplication), enabling efficient implementation of neural networks.

2. **Real-world Data Representation**  
   Complex data types like images, audio, text, and video can be naturally represented as tensors.

3. **Efficient Computations**  
   Tensors are optimized for hardware acceleration on GPUs and TPUs — a must for modern deep learning training.

---

## Tensors in Deep Learning

Tensors are involved in every step of building and training a neural network:

1. **Input Data** – Training data (e.g., images, text) is converted into tensors.
2. **Model Parameters** – Weights and biases of neural networks are stored as tensors.
3. **Operations** – Neural networks perform computations like matrix multiplication, dot products, and broadcasting — all using tensors.
4. **Forward & Backward Passes** – During training, data flows through layers as tensors. Gradients, used in optimization, are also stored and updated as tensors.

```python
import torch

# Example: Create a 2D tensor (matrix)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)  # Output: torch.Size([2, 2])
```

---

## Common Tensor Operations

- `reshape()`, `view()` – Change tensor shape without changing data
- `permute()` – Rearrange dimensions
- `matmul()` – Matrix multiplication
- `sum()`, `mean()`, `max()` – Aggregations
- `unsqueeze()` / `squeeze()` – Add or remove dimensions
- `to(device)` – Move tensor to CPU/GPU

---
