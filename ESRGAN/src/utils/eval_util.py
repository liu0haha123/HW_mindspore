import cv2
import sys
import time
import numpy as np
from absl import logging
import numpy as np

from src.utils.matlab_functions import bgr2ycbcr


def reorder_image(img, input_order="HWC"):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ["HWC", "CHW"]:
        raise ValueError(
            f"Wrong input_order {input_order}. Supported input_orders are "
            "'HWC' and 'CHW'"
        )
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == "CHW":
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.0


###############################################################################
#   These processing code is copied and modified from official implement:     #
#    https://github.com/open-mmlab/mmsr                                       #
###############################################################################
def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC RBG [0,1]
    # output: HWC RBG [0,1] w/o round
    # (Modified from
    #  https://github.com/open-mmlab/mmsr/blob/master/codes/data/util.py)
    in_H, in_W, in_C = img.shape

    _, out_H, out_W = in_C, np.ceil(in_H * scale), np.ceil(in_W * scale)
    out_H, out_W = out_H.astype(np.int64), out_W.astype(np.int64)
    kernel_width = 4
    kernel = "cubic"

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = _calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing
    )
    weights_W, indices_W, sym_len_Ws, sym_len_We = _calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing
    )
    # process H dimension
    # symmetric copying
    img_aug = np.zeros(((in_H + sym_len_Hs + sym_len_He), in_W, in_C))
    img_aug[sym_len_Hs : sym_len_Hs + in_H] = img

    sym_patch = img[:sym_len_Hs, :, :]
    sym_patch_inv = sym_patch[::-1]
    img_aug[0:sym_len_Hs] = sym_patch_inv

    sym_patch = img[-sym_len_He:, :, :]
    sym_patch_inv = sym_patch[::-1]
    img_aug[sym_len_Hs + in_H : sym_len_Hs + in_H + sym_len_He] = sym_patch_inv

    out_1 = np.zeros((out_H, in_W, in_C))
    kernel_width = weights_H.shape[1]
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = weights_H[i].dot(
            img_aug[idx : idx + kernel_width, :, 0].transpose(0, 1)
        )
        out_1[i, :, 1] = weights_H[i].dot(
            img_aug[idx : idx + kernel_width, :, 1].transpose(0, 1)
        )
        out_1[i, :, 2] = weights_H[i].dot(
            img_aug[idx : idx + kernel_width, :, 2].transpose(0, 1)
        )

    # process W dimension
    # symmetric copying
    out_1_aug = np.zeros((out_H, in_W + sym_len_Ws + sym_len_We, in_C))
    out_1_aug[:, sym_len_Ws : sym_len_Ws + in_W] = out_1

    sym_patch = out_1[:, :sym_len_Ws, :]
    sym_patch_inv = sym_patch[:, ::-1]
    out_1_aug[:, 0:sym_len_Ws] = sym_patch_inv

    sym_patch = out_1[:, -sym_len_We:, :]
    sym_patch_inv = sym_patch[:, ::-1]
    out_1_aug[:, sym_len_Ws + in_W : sym_len_Ws + in_W + sym_len_We] = sym_patch_inv

    out_2 = np.zeros((out_H, out_W, in_C))
    kernel_width = weights_W.shape[1]
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx : idx + kernel_width, 0].dot(weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx : idx + kernel_width, 1].dot(weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx : idx + kernel_width, 2].dot(weights_W[i])

    return out_2.clip(0, 255)


def _cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).astype(np.float64)) + (
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
    ) * (((absx > 1) * (absx <= 2)).astype(np.float64))


def _calculate_weights_indices(
    in_length, out_length, scale, kernel, kernel_width, antialiasing
):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias
        # larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = np.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = np.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = (np.ceil(kernel_width) + 2).astype(np.int32)

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.reshape(int(out_length), 1).repeat(P, axis=1) + np.linspace(
        0, P - 1, P
    ).reshape(1, int(P)).repeat(out_length, axis=0)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.reshape(int(out_length), 1).repeat(P, axis=1) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * _cubic(distance_to_center * scale)
    else:
        weights = _cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = np.sum(weights, 1).reshape(int(out_length), 1)
    weights = weights / weights_sum.repeat(P, axis=1)

    # If a column in weights is all zero, get rid of it. only consider the
    # first and last column.
    weights_zero_tmp = np.sum((weights == 0), 0)
    if not np.isclose(weights_zero_tmp[0], 0, rtol=1e-6):
        indices = indices[:, 1 : 1 + int(P) - 2]
        weights = weights[:, 1 : 1 + int(P) - 2]
    if not np.isclose(weights_zero_tmp[-1], 0, rtol=1e-6):
        indices = indices[:, 0 : 0 + int(P) - 2]
        weights = weights[:, 0 : 0 + int(P) - 2]
    weights = weights.copy()
    indices = indices.copy()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def _ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return _ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for _ in range(3):
                ssims.append(_ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return _ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def rgb2ycbcr(img, only_y=True):
    """Convert rgb to ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.0
    img = img[:, :, ::-1]

    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(
            img,
            [
                [24.966, 112.0, -18.214],
                [128.553, -74.203, -93.786],
                [65.481, -37.797, 112.0],
            ],
        ) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)