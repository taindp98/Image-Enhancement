import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
from matplotlib import pyplot as plt
from image_enhancement.utils import add_noisy, transform_wavelet, compute_psnr, compute_ssim


def divide_patches(image: np.ndarray, patch_size: tuple = (7, 7)) -> np.ndarray:
    """
    Divides the input image into smaller patches of the specified size.

    Args:
        image (np.ndarray): The input image to be divided into patches.
        patch_size (tuple): A tuple specifying the size of each patch (height, width).

    Returns:
        np.ndarray: An array where each row corresponds to a flattened patch from the image.
    """
    patches_array = np.lib.stride_tricks.sliding_window_view(image, patch_size)
    patches_array_rot = np.moveaxis(patches_array, 0, 1)
    training_region = patches_array_rot.reshape(
        (
            patches_array_rot.shape[0] * patches_array_rot.shape[1],
            patches_array_rot.shape[2] * patches_array_rot.shape[3],
        )
    )
    return training_region


def define_centroid_patches(
    image: np.ndarray, patch_size: int = 7, stride_overlap: int = 1
) -> tuple:
    """
    Defines the centroid patches for an input image based on the specified patch size and stride overlap.

    Args:
        image (np.ndarray): The input image for which to define centroid patches.
        patch_size (int): The size of each patch (assumed to be square).
        stride_overlap (int): The stride (or overlap) between patches.

    Returns:
        tuple: A tuple containing:
            - centroids_map_index (np.ndarray): An index matrix of the centroids.
            - centroids_width (int): The number of centroids along the width of the image.
            - centroids_height (int): The number of centroids along the height of the image.
    """
    # Calculate the number of centroids in width and height directions
    centroids_width = int((image.shape[0] - patch_size) / stride_overlap) + 1
    centroids_height = int((image.shape[1] - patch_size) / stride_overlap) + 1
    total_num_centroids = centroids_width * centroids_height

    # Define the index matrix of the centroids array
    centroids_map_index = np.arange(total_num_centroids)
    centroids_map_index = centroids_map_index.reshape((centroids_height, centroids_width)).T

    return centroids_map_index, centroids_width, centroids_height


def get_principal_components(
    centroids_map_index: np.ndarray,
    training_region: np.ndarray,
    current_row: int,
    current_col: int,
    current_centroid_idx: int,
    train_region_size: int = 21,
    n_components: int = 50,
) -> np.ndarray:
    """
    Selects the principal components based on the distance from the current centroid.

    Args:
        centroids_map_index (np.ndarray): Index matrix of the centroids.
        training_region (np.ndarray): Array representing the training region patches.
        current_row (int): Current row index of the centroid.
        current_col (int): Current column index of the centroid.
        current_centroid_idx (int): Index of the current centroid in the training region.
        train_region_size (int): Size of the training region to consider around the current centroid.
        n_components (int): Number of principal components to select.

    Returns:
        np.ndarray: Indices of the principal components selected based on the distance.
    """
    [centroids_width, centroids_height] = centroids_map_index.shape

    rmin = max(current_row - train_region_size - 1, 0)
    rmax = min(current_row + train_region_size, centroids_width)
    cmin = max(current_col - train_region_size - 1, 0)
    cmax = min(current_col + train_region_size, centroids_height)

    idx = centroids_map_index[rmin:rmax, cmin:cmax].T.flatten()

    # Extract the training set and denoise region
    training_set = training_region[idx, :]
    denoise_region = training_region[current_centroid_idx, :]

    # Calculate the distance to select principal components
    init_distance = (training_set[:, 0] - denoise_region[0]) ** 2
    init_distance = init_distance.reshape((init_distance.shape[0], 1))

    for k in range(1, training_region.shape[1]):
        partial_distance = (training_set[:, k] - denoise_region[k]) ** 2
        partial_distance = partial_distance.reshape((partial_distance.shape[0], 1))
        init_distance += partial_distance

    components_distance = init_distance / training_region.shape[1]
    components_distance_sort = np.argsort(components_distance, axis=0)

    pc_index = idx[components_distance_sort[0:n_components]]

    return pc_index


def pca_transform(x: np.ndarray) -> tuple:
    """
    Transforms the input data from the spatial domain to the PCA domain.

    Args:
        x (np.ndarray): An MxN matrix where M is the number of dimensions and N is the number of trials.

    Returns:
        tuple: A tuple containing:
            - transform_coef (np.ndarray): The transformed data in the PCA domain.
            - transform_mat (np.ndarray): The transformation matrix.
            - variance (np.ndarray): The variance vector.
            - mean_x (np.ndarray): The mean vector of the input data.
    """
    m, n = x.shape

    # Shift data to the center (subtract mean from every dimension)
    mean_x = np.mean(x, axis=1).reshape((m, 1))
    x_centered = x - mean_x

    # Compute covariance matrix for centered data
    covar_x = np.matmul(x_centered, x_centered.T) / (n - 1)

    # Find eigenvalues and eigenvectors of the covariance matrix
    variance, transform_mat = np.linalg.eig(covar_x)

    # Sort eigenvectors by corresponding eigenvalues in descending order
    sorted_indices = np.argsort(-variance)
    variance = variance[sorted_indices]
    transform_mat = transform_mat[:, sorted_indices]

    # Transform the data
    transform_mat = transform_mat.T
    transform_coef = np.matmul(transform_mat, x_centered)

    return transform_coef, transform_mat, variance, mean_x


def adapt_pc_denoising(
    img_noise: np.ndarray,
    patch_size: int = 7,
    stride_overlap: int = 1,
    train_region_size: int = 21,
    n_components: int = 50,
) -> tuple:
    """
    Performs adaptive principal component denoising on the input noisy image.

    Args:
        img_noise (np.ndarray): The input noisy image.
        patch_size (int): The size of each patch.
        stride_overlap (int): The stride (or overlap) between patches.
        train_region_size (int): The size of the training region around each centroid.
        n_components (int): The number of principal components to select.

    Returns:
        tuple: A tuple containing:
            - img_denoise (np.ndarray): The denoised image.
            - compare_psnr (tuple): The PSNR of the noisy and denoised images compared to the original image.
            - compare_ssim (tuple): The SSIM of the noisy and denoised images compared to the original image.
    """
    # Transform the noisy image to the wavelet domain and estimate noise variance
    y_coeffs = transform_wavelet(img_noise)
    var_noise_est = np.median(np.abs(y_coeffs[-1][-1])) / 0.6745

    # Define centroids and divide the image into patches
    centroids_map_index, centroids_width, centroids_height = define_centroid_patches(
        image=img_noise, patch_size=patch_size, stride_overlap=stride_overlap
    )
    training_region = divide_patches(img_noise)
    denoise_set = np.zeros(training_region.T.shape)

    # Training phase
    print("*" * 10, "Training", "*" * 10)
    with tqdm(total=centroids_width * centroids_height) as pbar:
        for i in range(centroids_width):
            for j in range(centroids_height):
                current_row = i
                current_col = j
                current_centroid_idx = current_col * centroids_width + current_row

                # Get principal components
                pc_idx = get_principal_components(
                    centroids_map_index=centroids_map_index,
                    training_region=training_region,
                    current_row=current_row,
                    current_col=current_col,
                    current_centroid_idx=current_centroid_idx,
                    train_region_size=train_region_size,
                    n_components=n_components,
                )

                pc_idx = pc_idx.flatten()

                # Perform PCA transformation
                transform_coef, transform_mat, variance, mean_x = pca_transform(
                    training_region.T[:, pc_idx]
                )

                # Calculate distance to select principal components
                py = np.mean(transform_coef**2, axis=1).reshape((-1, 1))
                px = np.maximum(np.zeros(py.shape), py - var_noise_est**2)
                wei = px / py

                # PCA inverse transformation
                trans_coeff_est = (transform_coef[:, 0] * wei.T).T
                denoise_est = np.matmul(transform_mat.T, trans_coeff_est)
                denoise_set[:, current_centroid_idx] = (denoise_est + mean_x)[:, 0]
                pbar.update(1)

    # Reconstruction phase
    print("*" * 10, "Reconstructing", "*" * 10)
    img_recon = np.zeros(img_noise.shape)
    img_wei = np.zeros(img_noise.shape)
    row_idx = np.arange(0, centroids_width)
    col_idx = np.arange(0, centroids_height)

    k = 0
    for i in range(patch_size):
        for j in range(patch_size):
            rv, cv = np.meshgrid(row_idx + i, col_idx + j)
            img_recon[rv, cv] += (denoise_set[k, :].T).reshape((centroids_width, centroids_height))
            img_wei[rv, cv] += 1
            k += 1
    img_denoise = img_recon / img_wei

    # Calculate PSNR and SSIM for comparison
    compare_psnr = (compute_psnr(img_gray, img_noise), compute_psnr(img_gray, img_denoise))
    compare_ssim = (compute_ssim(img_gray, img_noise), compute_ssim(img_gray, img_denoise))

    return img_denoise, compare_psnr, compare_ssim


if __name__ == "__main__":
    ROOT = "samples"
    OUTDIR = "outputs"

    img_path = os.path.join(ROOT, "lena512.bmp")
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_origin = img_gray

    noise_var = 25**2

    img_noise, noise = add_noisy(img_gray, var=noise_var)
    img_denoise, compare_psnr, compare_ssim = adapt_pc_denoising(
        img_noise, patch_size=7, stride_overlap=1, train_region_size=21, n_components=50
    )
    img_denoise_format_int = (
        ((img_denoise - img_denoise.min()) / (img_denoise.max() - img_denoise.min())) * 255.9
    ).astype(np.uint8)
    img_denoise_array = Image.fromarray(img_denoise_format_int)

    denoise_img_path = os.path.join(OUTDIR, "denoise.png")

    plt.figure(figsize=(10, 8), dpi=80)
    plt.subplot(1, 3, 1)
    plt.imshow(img_origin, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    img_noise, noise = add_noisy(img_gray, var=noise_var)
    plt.imshow(img_noise, cmap="gray")
    plt.title("Image noisy")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img_denoise, cmap="gray")
    plt.title("Image denoise")
    plt.axis("off")

    plt.savefig(denoise_img_path)
