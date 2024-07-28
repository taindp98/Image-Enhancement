import numpy as np
import pywt
import cv2


def add_noisy(img: np.ndarray, var: float = None) -> tuple:
    """
    Create a noise array with the same size as the input image and add it to the image.

    Args:
        img (np.ndarray): Input image.
        var (float, optional): Variance of the Gaussian noise. Default is 25.

    Returns:
        tuple: A tuple containing the noisy image and the noise array.
    """
    row, col = img.shape
    mean = 0
    if var is None:
        var = 25
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = img + gauss
    return noisy, gauss


def transform_wavelet(spatial: np.ndarray, algo: str = "db2") -> tuple:
    """
    Apply discrete wavelet transform to the input image.

    Args:
        spatial (np.ndarray): Input image.
        algo (str, optional): Wavelet to use. Default is "db2".

    Returns:
        tuple: Coefficients of the wavelet transform.
    """
    coeffs2 = pywt.dwt2(spatial, algo)
    return coeffs2


def compute_psnr(img_origin: np.ndarray, img_denoise: np.ndarray) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img_origin (np.ndarray): Original image.
        img_denoise (np.ndarray): Denoised image.

    Returns:
        float: PSNR value.
    """
    mse = np.mean((img_origin - img_denoise) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    Args:
        img1 (np.ndarray): First input image.
        img2 (np.ndarray): Second input image.

    Returns:
        float: SSIM value.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate the Structural Similarity Index (SSIM) between two images.

    Args:
        img1 (np.ndarray): First input image.
        img2 (np.ndarray): Second input image.

    Returns:
        float: SSIM value.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")
