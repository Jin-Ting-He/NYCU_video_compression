from PIL import Image
import numpy as np

# Load image
image = Image.open('lena.png')
pixels = np.array(image)

# Prepare empty arrays for the grayscale images
R_image = np.empty(pixels.shape[:2], dtype=np.uint8)
G_image = np.empty(pixels.shape[:2], dtype=np.uint8)
B_image = np.empty(pixels.shape[:2], dtype=np.uint8)
Y_image = np.empty(pixels.shape[:2], dtype=np.uint8)
U_image = np.empty(pixels.shape[:2], dtype=np.uint8)
V_image = np.empty(pixels.shape[:2], dtype=np.uint8)
Cb_image = np.empty(pixels.shape[:2], dtype=np.uint8)
Cr_image = np.empty(pixels.shape[:2], dtype=np.uint8)

# Calculate R, G, B, Y, U, V, Cb, Cr for each pixel
for i in range(pixels.shape[0]):
    for j in range(pixels.shape[1]):
        R, G, B = pixels[i, j][:3]

        # Y, U, V calculations
        Y = int(0.299 * R + 0.587 * G + 0.114 * B)
        U = int(-0.169 * R - 0.331 * G + 0.5 * B + 128)
        V = int(0.5 * R - 0.419 * G - 0.081 * B + 128)
        
        # YCbCr calculations
        Cb = int(128 - 0.168736 * R - 0.331264 * G + 0.5 * B)
        Cr = int(128 + 0.5 * R - 0.418688 * G - 0.081312 * B)
        
        # Store values
        R_image[i, j] = R
        G_image[i, j] = G
        B_image[i, j] = B
        Y_image[i, j] = Y
        U_image[i, j] = U
        V_image[i, j] = V
        Cb_image[i, j] = Cb
        Cr_image[i, j] = Cr
    
    # Show progress
    print("Completing Row",i,"/",pixels.shape[0])

# Save images
Image.fromarray(R_image).save('output/R_image.png')
Image.fromarray(G_image).save('output/G_image.png')
Image.fromarray(B_image).save('output/B_image.png')
Image.fromarray(Y_image).save('output/Y_image.png')
Image.fromarray(U_image).save('output/U_image.png')
Image.fromarray(V_image).save('output/V_image.png')
Image.fromarray(Cb_image).save('output/Cb_image.png')
Image.fromarray(Cr_image).save('output/Cr_image.png')
