import cv2
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio
# Evaluate PSNR
def psnr(original, reconstructed):
    # # 将图像转换为 numpy 数组
    # original = np.array(original, dtype=np.float64)
    # reconstructed = np.array(reconstructed, dtype=np.float64)
    mse = np.mean((original - reconstructed) ** 2)
    max_pixel = 255.0
    psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_val

def full_search(image1, image2, block_size, search_range):
    st = time.time()
    height, width = image1.shape
    # Initialize the motion vectors and the predicted image
    motion_vectors = np.zeros((height // block_size, width // block_size, 2), dtype=np.int32) 
    predicted_image = np.zeros_like(image1)
    # Full Search Block Matching Algorithm
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            ref_block = image2[i:i+block_size, j:j+block_size]
            best_match = np.inf
            m = [i, j]
            for x in range(-search_range, search_range + 1):
                for y in range(-search_range, search_range + 1):
                    x_search = i + x
                    y_search = j + y

                    # Boundary check
                    if x_search < 0 or y_search < 0 or x_search + block_size > height or y_search + block_size > width:
                        continue
                    
                    candidate_block = image1[x_search:x_search+block_size, y_search:y_search+block_size]
                    
                    error = np.mean((ref_block - candidate_block) ** 2)

                    if error < best_match:
                        best_match = error
                        dx, dy = x, y
                        m = [x_search, y_search]

            # Save the motion vector
            motion_vectors[i // block_size, j // block_size] = [dx, dy]
            
            # Reconstruct the block in the predicted image
            predicted_image[i:i+block_size, j:j+block_size] = image1[m[0]:m[0]+block_size, m[1]:m[1]+block_size]

    predictimg = motion_compernsation(image1, motion_vectors, block_size)
    # Calculate the residual image
    residual_image = cv2.absdiff(image2, predictimg)
    # Save the predicted and residual images
    cv2.imwrite('output/predimg_blocksize'+str(block_size)+'_searchrange'+str(search_range)+'_full'+'.png', predicted_image)
    cv2.imwrite('output/residual_image'+str(block_size)+'_searchrange'+str(search_range)+'_full'+'.png', residual_image)
    PSNR = psnr(predictimg, image2)
    et = time.time()
    time_cost = et - st
    return round(PSNR, 2),round(time_cost, 2)
def motion_compernsation(img, motion_vector, block_size):
    height, width = img.shape
    result_img = np.zeros_like(img)
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            motion = motion_vector[i//block_size, j//block_size]
            i_s, i_e = i + motion[0], i + motion[0] + block_size
            j_s, j_e = j + motion[1], j + motion[1] + block_size
            result_img[i:i+block_size,j:j+block_size] = img[i_s:i_e,j_s:j_e]
    return result_img
def three_step_search(image1, image2, block_size, search_range):
    st = time.time()
    height, width = image1.shape
    # Initialize the motion vectors and the predicted image
    motion_vectors = np.zeros((height // block_size, width // block_size, 2), dtype=np.int32)
    predicted_image = np.zeros_like(image1)
    k = int(np.log2(search_range))
    step_list = [2**i for i in range(k-1, -1, -1)]
    # Full Search Block Matching Algorithm
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            ref_block = image2[i:i+block_size, j:j+block_size]
            local_search_range = search_range
            best_match = np.inf
            m = [i, j]
            for step in step_list:
                for x in range(-local_search_range, local_search_range+1, step):
                    for y in range(-local_search_range, local_search_range+1, step):
                        x_search = m[0] + x
                        y_search = m[1] + y

                        # Boundary check
                        if x_search < 0 or y_search < 0 or x_search + block_size > height or y_search + block_size > width:
                            continue
                        
                        candidate_block = image1[x_search:x_search+block_size, y_search:y_search+block_size]
                        
                        error = np.mean((ref_block - candidate_block) ** 2)

                        if error < best_match:
                            best_match = error
                            m = [x_search, y_search]
                            dx, dy = x, y
                local_search_range = local_search_range // 2

            # Save the motion vector
            motion_vectors[i // block_size, j // block_size] = [dx, dy]
            # Reconstruct the block in the predicted image
            predicted_image[i:i+block_size, j:j+block_size] = image1[m[0]:m[0]+block_size, m[1]:m[1]+block_size]
    
    # predictimg = motion_compernsation(image1, motion_vectors, block_size)
    # Calculate the residual image
    residual_image = cv2.absdiff(image2, predicted_image)

    # Save the predicted and residual images
    cv2.imwrite('output/predimg_blocksize'+str(block_size)+'_searchrange'+str(search_range)+'_threestep'+'.png', predicted_image)
    cv2.imwrite('output/residual_image'+str(block_size)+'_searchrange'+str(search_range)+'_threestep'+'.png', residual_image)
    print("pakage psnr: ", peak_signal_noise_ratio(predicted_image, image2))
    PSNR = psnr(predicted_image, image2)
    et = time.time()
    time_cost = et - st
    return round(PSNR, 2), round(time_cost, 2)

if __name__ == "__main__":
    path_to_first_image = 'one_gray.png'
    path_to_second_image = 'two_gray.png'
    # Load the images in grayscale
    image1 = cv2.imread(path_to_first_image, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(path_to_second_image, cv2.IMREAD_GRAYSCALE)
    # Parameters for block matching
    block_size = 8

    # For Search Range 8 Experiments
    print("########### Search Range +-8 ############")
    search_range = 8
    psnr_full, time_cost_full = full_search(image1, image2, block_size, search_range)
    psnr_three_step, time_cost_three_step = three_step_search(image1, image2, block_size, search_range)

    print("full search:       ","PSNR: ",psnr_full,"Runtime: ", time_cost_full)
    print("three step search: ","PSNR: ",psnr_three_step,"Runtime: ", time_cost_three_step)

    # For Search Range 16 Experiments
    print("########### Search Range +-16 ############")
    search_range = 16
    psnr_full, time_cost_full = full_search(image1, image2, block_size, search_range)
    psnr_three_step, time_cost_three_step = three_step_search(image1, image2, block_size, search_range)

    print("full search:       ","PSNR: ",psnr_full,"Runtime: ", time_cost_full)
    print("three step search: ","PSNR: ",psnr_three_step,"Runtime: ", time_cost_three_step)

    # For Search Range 32 Experiments
    print("########### Search Range +-32 ############")
    search_range = 32
    psnr_full, time_cost_full = full_search(image1, image2, block_size, search_range)
    psnr_three_step, time_cost_three_step = three_step_search(image1, image2, block_size, search_range)

    print("full search:       ","PSNR: ",psnr_full,"Runtime: ", time_cost_full)
    print("three step search: ","PSNR: ",psnr_three_step,"Runtime: ", time_cost_three_step)

