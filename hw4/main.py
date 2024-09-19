import numpy as np
import cv2
import os
import sys

# Evaluate PSNR
def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    max_pixel = 255.0
    psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
    return round(psnr_val,3)

def get_size(filename):
    stat = os.stat(filename)
    size=stat.st_size
    return size

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

# Run-Length Encoding Functions
def run_length_encode(arr):
    counts = []
    values = []
    encoded = []
    prev_value = arr[0]
    count = 1
    for elem in arr[1:]:
        if elem == prev_value:
            count += 1
        else:
            counts.append(count)
            values.append(prev_value)
            encoded.append((count, prev_value))
            count = 1
            prev_value = elem
    counts.append(count)
    values.append(prev_value)
    encoded.append((count, prev_value))
    encoded = np.array(encoded).astype(np.int16)
    return np.array(values), np.array(counts), encoded

def run_length_decode(values, counts):
    return np.repeat(values, counts)

# DCT and IDCT Functions
def block_dct(block):
    return cv2.dct(np.float32(block))

def block_idct(block):
    return cv2.idct(block)

# Quantization Functions
def quantize(block, dc_bit=16, ac_bit=8):
    # print(block)
    # setting the quantize factor
    dc_quantized_factor = (np.max(block) - np.min(block)) / (2**dc_bit)
    ac_quantized_factor = (np.max(block) - np.min(block)) / (2**ac_bit)
    quantized_block = np.zeros_like(block, dtype=np.int32)
    quantized_block[0, 0] = np.round(block[0, 0] / dc_quantized_factor) * dc_quantized_factor
    quantized_block[0:7, 1:7] =np.round(block[0:7, 1:7] / ac_quantized_factor) * ac_quantized_factor
    quantized_block[1, 0:7] = np.round(block[1, 0:7] / ac_quantized_factor) * ac_quantized_factor
    # print(quantized_block)
    return quantized_block

# Zigzag Scan Functions
def zigzag_indices(rows, cols):
    indices = [(i, j) for i in range(rows) for j in range(cols)]
    return sorted(indices, key=lambda x: (x[0]+x[1], -x[1] if (x[0]+x[1]) % 2 else x[1]))

def zigzag_scan(block):
    return np.array([block[i[0], i[1]] for i in zigzag_indices(8, 8)])

def de_zigzag_scan(zigzag):
    block = np.zeros((8,8))
    c = 0
    for i in zigzag_indices(8, 8):
        block[i[0], i[1]] = zigzag[c]
        c += 1
    return block
# Image Processing Functions
def process_image_blocks(image, block_size=8):
    h, w = image.shape
    processed_image = np.zeros_like(image, dtype=np.float32)
    encoded = np.array([[0, 0]])
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):                
            block = image[i:i+block_size, j:j+block_size]
            block = np.asarray(block, dtype=np.float32)
            dct_block = dct_1d(block)
            quantized_block = quantize(dct_block)
            zigzag = zigzag_scan(quantized_block)
            values, counts, single_encoded = run_length_encode(zigzag)
            encoded = np.append(encoded, single_encoded, axis = 0)
            # Decoding and inverse operations
            decoded_zigzag = run_length_decode(values, counts)
            de_zigzag_block = de_zigzag_scan(decoded_zigzag)
            idct_block = idct_1d(de_zigzag_block)
            processed_image[i:i+block_size, j:j+block_size] = idct_block
    earr = np.array(encoded)
    # np.savez("encoded.npz", earr)
    cv2.imwrite("encoded.png", earr)
    return processed_image, earr

# Load Image
image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image file not found.")

# Ensure the image dimensions are multiples of 8
height, width = image.shape
image = image[:height - height % 8, :width - width % 8]

# Process Image
processed_image, earr = process_image_blocks(image)

# Convert processed image back to uint8
processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)

# Save reconstruct image
cv2.imwrite("reconstructed_img.png", processed_image)
cv2.imwrite("lena_gray.png", image)

print("File size of encoded file: ", get_size("encoded.png")," bytes")
print("File size of original lena.png: ",get_size("lena_gray.png")," bytes")
print("PSNR of lena.png and reconstructed img: ", psnr(processed_image, image))

