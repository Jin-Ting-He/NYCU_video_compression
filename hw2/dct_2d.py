import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

# Evaluate PSNR
def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    max_pixel = 255.0
    psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_val

def compute_dct_value(u, v, M, N, image):
    # Create cosine matrices
    x_vals = np.arange(M).reshape(-1, 1)  # Column vector
    y_vals = np.arange(N).reshape(1, -1)  # Row vector

    cos_x = np.cos((2 * x_vals + 1) * u * np.pi / (2 * M))
    cos_y = np.cos((2 * y_vals + 1) * v * np.pi / (2 * N))

    # Element-wise multiplication and summation
    sum_val = np.sum(image * cos_x * cos_y)

    alpha_u = np.sqrt(1/M) if u == 0 else np.sqrt(2/M)
    alpha_v = np.sqrt(1/N) if v == 0 else np.sqrt(2/N)
    return alpha_u * alpha_v * sum_val
def dct_2d(image):
    M, N = image.shape
    image_dct = np.zeros((M, N))
    for u in range(M):
        print("dct row ",u,"/",M)
        for v in range(N):
            image_dct[u, v] = compute_dct_value(u, v, M, N, image)
    return image_dct

def compute_idct_value(x, y, M, N, image_dct):
    # Create alpha matrices
    alpha_u_vals = np.where(np.arange(M) == 0, np.sqrt(1/M), np.sqrt(2/M)).reshape(-1, 1)
    alpha_v_vals = np.where(np.arange(N) == 0, np.sqrt(1/N), np.sqrt(2/N)).reshape(1, -1)

    # Create cosine matrices
    u_vals = np.arange(M).reshape(-1, 1)  # Column vector
    v_vals = np.arange(N).reshape(1, -1)  # Row vector

    cos_u = np.cos((2 * x + 1) * u_vals * np.pi / (2 * M))
    cos_v = np.cos((2 * y + 1) * v_vals * np.pi / (2 * N))

    # Element-wise multiplication and summation
    sum_val = np.sum(alpha_u_vals * alpha_v_vals * image_dct * cos_u * cos_v)
    return sum_val
def idct_2d(image_dct):
    M, N = image_dct.shape
    image_reconstructed = np.zeros((M, N))
    for x in range(M):
        print("idct row: ",x,"/",M)
        for y in range(N):
            image_reconstructed[x, y] = compute_idct_value(x, y, M, N, image_dct)
    return image_reconstructed

if __name__ =="__main__":
    # Load and preprocess the image
    img = Image.open("lena.png").convert("L")
    img_data = np.asarray(img, dtype=np.float32)
    # Compute the 2D-DCT of the image
    st = time.time()
    dct_coefficients = dct_2d(img_data)
    et = time.time()
    print('time cost: ',et-st)
    # Reconstruct the image using the 2D-IDCT
    reconstructed_image = idct_2d(dct_coefficients)
    psnr_value = psnr(img_data, reconstructed_image)
    print(f"PSNR value: {psnr_value:.2f} dB")
    # Visualize the coefficients in the log domain
    plt.imshow(np.log1p(np.abs(dct_coefficients)), cmap="gray")
    plt.title("DCT Coefficients in Log Domain")
    plt.colorbar()
    plt.show()
