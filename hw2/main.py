from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dct_1d import *
from dct_2d import *
import time

def run_dct_idct_1d(img_data):
    st = time.time()
    dct_coeffs = dct_1d(img_data)
    et = time.time()
    reconstructed_img_data = idct_1d(dct_coeffs)
    psnr_value = psnr(img_data, reconstructed_img_data)
    return et-st, psnr_value, dct_coeffs, reconstructed_img_data
def run_dct_idct_2d(img_data):
    st = time.time()
    dct_coeffs = dct_2d(img_data)
    et = time.time()
    reconstructed_img_data = idct_2d(dct_coeffs)
    psnr_value = psnr(img_data, reconstructed_img_data)
    return et-st, psnr_value, dct_coeffs, reconstructed_img_data
if __name__ =="__main__":
     # Load and preprocess the image
    img = Image.open("lena.png").convert("L")
    img_data = np.asarray(img, dtype=np.float32)

    # DCT, IDCT
    time_1d, psnr_1d, dct_img_1d, recons_img_1d = run_dct_idct_1d(img_data)
    time_2d, psnr_2d, dct_img_2d, recons_img_2d = run_dct_idct_2d(img_data)

    # Time cost, PSNR
    print("time cost 1D: ", time_1d," time cost 2D: ", time_2d)
    print("PSNR 1D: ",psnr_1d," PSNR 2D: ", psnr_2d)

    # Saving output image
    plt.imshow(np.log1p(np.abs(dct_img_1d)), cmap='gray')
    plt.title("1D-DCT Coefficients (Log Domain)")
    plt.colorbar()
    plt.savefig('1d-dct_coefficients.png')
    plt.gcf().clear()  # Clears the entire figure.

    plt.imshow(recons_img_1d, cmap='gray')
    plt.title("Reconstructed image(1D)")
    plt.savefig('reconstructed_image_1d.png')
    plt.gcf().clear()  # Clears the entire figure.

    plt.imshow(np.log1p(np.abs(dct_img_2d)), cmap='gray')
    plt.title("2D-DCT Coefficients (Log Domain)")
    plt.colorbar()
    plt.savefig('2d-dct_coefficients.png')
    plt.gcf().clear()  # Clears the entire figure.

    plt.imshow(recons_img_2d, cmap='gray')
    plt.title("Reconstructed image(2D)")
    plt.savefig('reconstructed_image_2d.png')
    plt.gcf().clear()  # Clears the entire figure.

