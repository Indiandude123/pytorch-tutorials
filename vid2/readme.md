# Tensors - multidimensional array, a data structure

# Dimension - how many directions does the tensor it span

- Scalars (0 dim tensor) 
- Vectors (1 dim tensor)
- Matrices (2 dim tensor)
- 3d Tensors - Adds a third dimension often used for stacking data.Like coloured images. An RGB image( eg 256x256): shape (256, 256, 3)
- 4d Tensors - Adds a batch size as an additional dimension to 3d data.Batches of RGB images. A batch of 32 images each of size 128x128 with 3 colour channels would have shape (32, 128, 128, 3)
- 5d Tensors - Adds a time dimension for data that changes over time. Video clips: represented as a sequence of frames where each frame is an rgb image. A batch of 10 video calips each with 16 frames of size 64x64 with 3 channels would have shape (10, 16, 64, 64, 3)

# Why are tensors useful?
1) Mathematical Operations - enable efficient mathematical computations necessary for neural network ops
2) Representation of real world data - data like images, audio, videos and text can be representted as tensors
3) Efficient computations - tensors are optimized for hardware acceleration allowing computations on GPUs or TPUs which are crucial for training DL models.

# Where are tensors used in DL?
1) Training data is stored as tensors
2) Learnable parameters of a neural network are stored as tensors
3) dot products, matrix multiplication and broadcasting all performed using tensors
4) during forward passes, tensors flow through the network. Gradients represented as tensors are calculated during the backward pass

