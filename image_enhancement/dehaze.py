import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import os


def extract_dark_channel(img: np.ndarray, local_size: int) -> np.ndarray:
    """
    Extract the dark channel of an image.

    Args:
        img (np.ndarray): Input image.
        local_size (int): Size of the local patch.

    Returns:
        np.ndarray: Dark channel of the image.
    """
    b, g, r = cv2.split(img)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (local_size, local_size))
    dark = cv2.erode(dc, kernel)
    return dark


def estimate_atmospheric(im: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """
    Estimate the atmospheric light in the image.

    Args:
        im (np.ndarray): Input image.
        dark (np.ndarray): Dark channel of the image.

    Returns:
        np.ndarray: Estimated atmospheric light.
    """
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx:]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def estimate_depth_map(im: np.ndarray, atmosphere: np.ndarray, local_size: int = 15, omega: float = 0.85) -> np.ndarray:
    """
    Estimate the depth map (transmission map) of an image.

    Args:
        im (np.ndarray): Input image.
        atmosphere (np.ndarray): Estimated atmospheric light.
        local_size (int, optional): Size of the local patch. Default is 15.
        omega (float, optional): Parameter to control the amount of haze removed. Default is 0.85.

    Returns:
        np.ndarray: Estimated depth map.
    """
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / atmosphere[0, ind]

    transmission = 1 - omega * extract_dark_channel(im3, local_size=local_size)
    return transmission



def guided_filter(im: np.ndarray, et: np.ndarray, radius: int = 60, eps: float = 0.0001) -> np.ndarray:
    """
    Apply a guided filter to the input image.

    Args:
        im (np.ndarray): Guidance image.
        et (np.ndarray): Input image to be filtered.
        radius (int, optional): Radius of the guided filter. Default is 60.
        eps (float, optional): Regularization parameter. Default is 0.0001.

    Returns:
        np.ndarray: Filtered image.
    """
    mean_img = cv2.boxFilter(im, cv2.CV_64F, (radius, radius))
    mean_p = cv2.boxFilter(et, cv2.CV_64F, (radius, radius))
    mean_img_p = cv2.boxFilter(im * et, cv2.CV_64F, (radius, radius))
    cov_img_p = mean_img_p - mean_img * mean_p

    mean_img_sq = cv2.boxFilter(im * im, cv2.CV_64F, (radius, radius))
    var = mean_img_sq - mean_img * mean_img

    a = cov_img_p / (var + eps)
    b = mean_p - a * mean_img

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))

    q = mean_a * im + mean_b
    return q

def refine_depth_map(img_path: str, et: np.ndarray, radius: int = 60, eps: float = 0.0001) -> np.ndarray:
    """
    Refine the depth map using a guided filter.

    Args:
        img_path (str): Path to the input image.
        et (np.ndarray): Initial estimated transmission map.
        radius (int, optional): Radius of the guided filter. Default is 60.
        eps (float, optional): Regularization parameter. Default is 0.0001.

    Returns:
        np.ndarray: Refined transmission map.
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    t = guided_filter(gray, et, radius=radius, eps=eps)

    return t


def recover(im: np.ndarray, t: np.ndarray, atmosphere: np.ndarray, tx: float = 0.1) -> np.ndarray:
    """
    Recover the haze-free image using the depth map and atmospheric light.

    Args:
        im (np.ndarray): Input hazy image.
        t (np.ndarray): Transmission map.
        atmosphere (np.ndarray): Estimated atmospheric light.
        tx (float, optional): Threshold for transmission map. Default is 0.1.

    Returns:
        np.ndarray: Recovered image.
    """
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - atmosphere[0, ind]) / t + atmosphere[0, ind]

    return res


if __name__ == "__main__":
    ROOT = "samples"
    OUTDIR = "outputs"

    img_path = os.path.join(ROOT, "aerial.png")
    img_haze = cv2.imread(img_path)

    local_size = 15
    omega = 0.85
    radius = 60
    eps = 1e-4
    tx = 0.1

    img_haze = img_haze / 255.0
    dark = extract_dark_channel(img_haze, local_size=local_size)
    atmosphere = estimate_atmospheric(img_haze, dark)
    te = estimate_depth_map(img_haze, atmosphere, local_size=local_size, omega=omega)
    t = refine_depth_map(img_path, te, radius=radius, eps=eps)
    img_dehaze = recover(img_haze, t, atmosphere, tx=tx)
    img_dehaze *= 255
    img_dehaze = img_dehaze.astype(int)

    plt.figure(figsize=(10, 8), dpi=80)
    plt.subplot(1, 2, 1)
    plt.imshow(img_haze)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_dehaze)
    plt.title("Image dehazed")
    plt.axis("off")

    dehazed_img_path = os.path.join(OUTDIR, "dehaze.png")
    plt.savefig(dehazed_img_path)
