# CNTK v2.4 Release Notes

## Highlights of this Release
- Move to CUDA9, cuDNN 7 and Visual Studio 2017.
- Removed Python 3.4 support.
- Support Volta GPU and FP16.
- Better ONNX support.
- CPU perf improvement.
- More OPs.

## OPs
- ``top_k`` operation: in the forward pass it computes the top (largest) k values and corresponding indices along the specified axis. In the backward pass the gradient is scattered to the top k elements (an element not in the top k gets a zero gradient).
- ``gather`` operation now supports an axis argument
- ``squeeze`` and ``expand_dims`` operations for easily removing and adding singleton axes
- ``zeros_like`` and ``ones_like`` operations. In many situations you can just rely on CNTK correctly broadcasting a simple 0 or 1 but sometimes you need the actual tensor.
- ``sum`` operation: Create a new Function instance that computes element-wise sum of input tensors.
- ``softsign`` operation: Create a new Function instance that computes the element-wise softsign of a input tensor.
- ``asinh`` operation: Create a new Function instance that computes the element-wise asinh of a input tensor.
- ``log_softmax`` operation: Create a new Function instance that computes the logsoftmax normalized values of a input tensor.
- ``hard_sigmoid`` operation: Create a new Function instance that computes the hard_sigmoid normalized values of a input tensor.
- ``element_and``, ``element_not``, ``element_or``, ``element_xor`` element-wise logic operations
- ``reduce_l1`` operation: Computes the L1 norm of the input tensor's element along the provided axes.
- ``reduce_l2`` operation: Computes the L2 norm of the input tensor's element along the provided axes..
- ``reduce_sum_square`` operation: Computes the sum square of the input tensor's element along the provided axes.
- ``image_scaler`` operation: Alteration of image by scaling its individual values.

## ONNX
- Improved ONNX support in CNTK.
- Update ONNX to the latest ONNX from https://github.com/onnx/onnx
- Fixed several bugs.

