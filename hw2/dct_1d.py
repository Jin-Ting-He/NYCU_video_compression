from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

# Generate DCT matrix
def dct_matrix_1d(N):
    mat = np.zeros((N, N))
    for k in range(N):
        if k == 0:
            coef = np.sqrt(1.0/N)
        else:
            coef = np.sqrt(2.0/N)
        
        for n in range(N):
            mat[k, n] = coef * np.cos(np.pi * (2*n + 1) * k / (2*N))
    return mat
def dct_1d(img_data):
    M, N = img_data.shape
    DCT_mat_M = dct_matrix_1d(M)
    DCT_mat_N = dct_matrix_1d(N)
    # Forward 2D DCT (using two 1D DCTs)
    return DCT_mat_M @ img_data @ DCT_mat_N.T

def idct_1d(dct_coeffs):
    M, N = dct_coeffs.shape
    DCT_mat_M = dct_matrix_1d(M)
    DCT_mat_N = dct_matrix_1d(N)
    # Inverse 2D DCT (using two 1D IDCTs)
    return DCT_mat_N.T @ dct_coeffs @ DCT_mat_M
# Evaluate PSNR
def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    max_pixel = 255.0
    psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_val


if __name__ =="__main__":
    # Load and preprocess the image
    img = Image.open("lena.png").convert("L")
    img_data = np.asarray(img, dtype=np.float32)
    st = time.time()
    dct_coeffs = dct_1d(img_data)
    et = time.time()
    print("time cost: ",et-st)
    # Visualize the DCT coefficients
    plt.imshow(np.log1p(np.abs(dct_coeffs)), cmap='gray')
    plt.title("DCT Coefficients (Log Domain)")
    plt.colorbar()
    plt.show()
    reconstructed_img_data = idct_1d(dct_coeffs)
    psnr_value = psnr(img_data, reconstructed_img_data)
    print(f"PSNR value: {psnr_value:.2f} dB")
