# SYSTEM IMPORTS
from typing import List, Tuple, Union
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..parameter import Parameter


class Conv2d(Module):
    def __init__(self,
                 num_kernels: int,
                 num_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int = 1,
                 padding: Union[str, Tuple[int, int]] = "valid") -> None:
        self.num_kernels: int = int(num_kernels)
        self.num_channels: int = int(num_channels)

        # self.kernel size will be [height, width]
        # however if the input is a scalar we assume height = width
        self.kernel_size: Tuple[int, int] = None
        if not isinstance(kernel_size, tuple):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            assert(len(kernel_size) == 2)
            self.kernel_size = tuple(kernel_size)

        self.stride: int = int(stride)
        self.padding: Union[str, Tuple[int, int]] = padding

        kernel_shapes: Tuple[int, int, int] = self.kernel_size + (self.num_channels,)
        self.W: Parameter = Parameter(np.random.randn(*kernel_shapes, num_kernels))
        self.b: Parameter = Parameter(np.random.randn(self.num_kernels))


    def compute_pad_buffer(self) -> Tuple[int, int]:
        padding_amount: Tuple[int, int] = None
        if isinstance(self.padding, str) and self.padding.lower() == "same":
            # gonna add kernel height and width amount (half to each side)
            padding_amount = ((self.W.shape[0]-1)//2, (self.W.shape[1]-1)//2)
        elif isinstance(self.padding, str) and self.padding.lower() == "valid":
            # no padding
            padding_amount = (0,0)
        elif isinstance(self.padding, tuple):
            padding_amount = self.padding
        else:
            raise RuntimeError("ERROR: unsupported padding type for padding %s"
                % (self.padding))

        return padding_amount


    # IMPORTANT: we expect X to have the following shape (and we will maintain this format)
    #            [batch_size, img_height, img_width, num_channels]
    def pad(self, X: np.ndarray) -> np.ndarray:
        # computes the new dimensions of the padded examples,
        # applies the padding (i.e. adds zeros around the examples)
        # and returns the padded examples
        padding_amount: Tuple[int, int] = self.compute_pad_buffer()

        # expect examples to have shape [batch_dim, height, width, num_channels]
        # thankfully numpy has a handy function for padding which we can use
        return np.pad(X,
                      [(0, 0),                                 # pad amounts for axis 0
                       (padding_amount[0], padding_amount[0]), # pad amounts for axis 1
                       (padding_amount[1], padding_amount[1]), # pad amounts for axis 2
                       (0, 0)],                                # pad amounts for axis 3
                      mode="constant",
                      constant_values=0.0)

    # this method computes the numpy array shape of the output volume (batch of images)
    def compute_output_shape(self, X_shape: Tuple[int, int, int, int]) -> Tuple[int, int,
                                                                                int, int]:
        num_examples, height, width, num_channels = X_shape
        kernel_height, kernel_width, kernel_channels, num_filters = self.W.val.shape

        out_shape: Tuple[int, int, int, int] = None
        if isinstance(self.padding, str) and self.padding.lower() == "same":
            out_shape = [num_examples, height, width, num_filters]
        elif isinstance(self.padding, str) and self.padding.lower() == "valid":
            out_shape = [num_examples, (height - kernel_height) // self.stride + 1,
                                       (width - kernel_width) // self.stride + 1,
                         num_filters]
        elif isinstance(self.padding, tuple):
            pad_height, pad_width = self.padding
            out_shape = [num_examples,
                         int(1 + (height + 2*pad_height - kernel_height) / self.stride),
                         int(1 + (width + 2*pad_width - kernel_width) / self.stride),
                         num_filters]
        else:
            raise RuntimeError("ERROR: unsupported padding type for padding %s"
                % (self.padding))
        return out_shape


    def forward(self, X: np.ndarray) -> np.ndarray:
        batch_dim, height, width, num_channels = X.shape
        kernel_height, kernel_width, kernel_channels, num_filters = self.W.val.shape

        # pad the input volume (i.e. batch of images)
        X_padded: np.ndarray = self.pad(X)

        # get the output shape and pre-allocated the volume
        out_shape: Tuple[int, int, int, int] = self.compute_output_shape(X.shape)
        _, out_height, out_width, _ = out_shape
        Y_hat: np.ndarray = np.zeros(out_shape, dtype=float)

        # for each pixel in the output volume, we're going to apply self.W.val
        # to the patch of the image (patch includes channels, but so does self.W.val)
        for i in range(out_height):
            for j in range(out_width):

                # get x,y coordinates in our image (const across the batch)
                height_start_idx: int = i*self.stride
                height_end_idx: int = height_start_idx + kernel_height

                width_start_idx: int = j*self.stride
                width_end_idx: int = width_start_idx + kernel_width

                # compute the pixel for each channel for each image in the batch
                Y_hat[:, i, j, :] = (X_padded[:, height_start_idx:height_end_idx,
                                              width_start_idx:width_end_idx, :,
                                              np.newaxis] *
                    self.W.val[np.newaxis, :, :, :]).sum(axis=(1,2,3))

        # need to add our bias to this. The reshaping is to make sure numpy adds correctly
        return Y_hat + self.b.val.reshape(1,1,1,-1)

    def backwards(self, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:
        batch_dim, height, width, num_channels = X.shape
        kernel_height, kernel_width, kernel_channels, num_filters = self.W.val.shape

        pad_height, pad_width = self.compute_pad_buffer()
        _, out_height, out_width, _ = dLoss_dModule.shape

        X_padded: np.ndarray = self.pad(X)

        dL_dX: np.ndarray = np.zeros_like(X_padded, dtype=float)

        # TODO: compute self.b.grad and self.W.grad
        #       remember: self.W.val was applied to a bunch of patches of the input image
        #       (which is in batch form). Therefore, you should try and take the derivative
        #       with respect to a speicific patch, then think about how you combine
        #       these little derivatives into the overall derivative for self.W.grad

        # slice out derivatives for elements that aren't part of the padding
        self.b.grad = dLoss_dModule.sum(axis=(0,1,2))
        for h in range(out_height):
            for w in range(out_width):
                height_start  = int(h*self.stride)
                height_end = int(height_start + kernel_height)
                width_start= int(w*self.stride)
                width_end = int(width_start + kernel_width)
                dL_dX[:,height_start:height_end,width_start:width_end,:] += np.sum(self.W.val[np.newaxis,:,:,:,:]*dLoss_dModule[:,h:h+1,w:w+1,np.newaxis,:],axis=4)
                self.W.grad += np.sum(X_padded[:,height_start:height_end,width_start:width_end,:,np.newaxis]* dLoss_dModule[:,h:h+1,w:w+1,np.newaxis,:],axis=0)
        return dL_dX[:, pad_height:pad_height+height, pad_width:pad_width+width, :]

    def parameters(self) -> List[Parameter]:
        return [self.W, self.b]

