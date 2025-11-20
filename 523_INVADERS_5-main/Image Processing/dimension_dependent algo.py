 #DIMENSION COMPARISON
import cv2
def compare_dimensions(image1_path, image2_path):
    # Load images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Extract dimensions
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape

    # Compare dimensions
    if width1 == width2 and height1 == height2:
        return "Both images have the same dimensions."
    elif width1 == width2 or height1 == height2:
        return "The images have the same aspect ratio but different dimensions."
    else:
        return "The images have different dimensions and aspect ratios."

if __name__ == "__main__":
    image1_path = "org_phonepe.jpg"
    image2_path = "fake_phonepe.jpg"

    result = compare_dimensions(image1_path, image2_path)
    print(result)
    
 # PIXEL BY PIXEL COMPARISON
import cv2
import numpy as np

def pixel_comparison(image1_path, image2_path):
    # Load the images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Ensure images have the same dimensions
    if img1.shape != img2.shape:
        raise ValueError("The dimensions of the two images must be the same.")

    # Compute the Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)
    
    return mse

if __name__ == "__main__":
    image1_path = "fake_phonepe.jpg"
    image2_path = "org_phonepe.jpg"

    mse_value = pixel_comparison(image1_path, image2_path)
    print(f"Mean Squared Error (MSE): {mse_value}")

    # Depending on the MSE value, you can make a decision on the similarity
    # Lower MSE values indicate more similar images.
    if mse_value > 20 : # Set a suitable threshold based on your application
        print("The images are similar.")
    else:
        print("The images are different.")   



# MEAN SQUARED ERROR (MSE) and PEAK SIGNAL-TO-NOICE RATIO (PSNR) 
    import cv2
import numpy as np

def compute_mse(image1, image2):
    """Compute the Mean Squared Error (MSE) between two images."""
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

def compute_psnr(mse, max_pixel=255.0):
    """Compute the Peak Signal-to-Noise Ratio (PSNR) given the MSE and maximum pixel value."""
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))

if __name__ == "__main__":
    # Load the images
    image1 = cv2.imread("fake_phonepe.jpg")
    image2 = cv2.imread("org_phonepe.jpg")

    # Ensure images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions.")

    # Compute MSE
    mse_value = compute_mse(image1, image2)
    print(f"Mean Squared Error (MSE): {20}")

# HISTOGRAM COMAPRISON
import cv2
import numpy as np

def compare_histograms(image1_path, image2_path):
    # Load images in grayscale mode
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Compute histograms
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # Compute Chi-Squared distance
    chi_squared_distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

    return chi_squared_distance

if __name__ == "__main__":
    image1_path = r"org_phonepe.jpg"
    image2_path = r"fake_phonepe.jpg"

    distance = compare_histograms(image1_path, image2_path)
    print(f"Chi-Squared Distance: {distance}")

    # Based on the distance value, you can make a decision on image similarity.
    # Lower values indicate more similar histograms.
    THRESHOLD = 0.1
    if distance < THRESHOLD:  # Set a suitable threshold based on your application
        print("The histograms are similar.")
    else:
        print("The histograms are different.")
        
        
# FEATURE MATCHING 
import cv2

def match_images(image1_path, image2_path, threshold=10):
    # Read images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default parameters
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Check if enough good matches are found
    if len(good_matches) >= 25.1:
        return "Features matched!"
    else:
        return "Features do not match."

if __name__ == "__main__":
    image1_path = "fake_phonepe.jpg"
    image2_path = "org_phonepe.jpg"

    result = match_images(image1_path, image2_path)
    print(result)


# TEXTURE ANAYLISIS
import cv2
import numpy as np

def compute_lbp_histogram(image):
    # Calculate LBP image
    lbp = np.zeros(image.shape, dtype=np.uint8)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            center = image[i, j]
            code = 0
            code |= (image[i-1, j-1] > center) << 7
            code |= (image[i-1, j] > center) << 6
            code |= (image[i-1, j+1] > center) << 5
            code |= (image[i, j+1] > center) << 4
            code |= (image[i+1, j+1] > center) << 3
            code |= (image[i+1, j] > center) << 2
            code |= (image[i+1, j-1] > center) << 1
            code |= (image[i, j-1] > center) << 0
            lbp[i, j] = code

    # Compute histogram
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    return hist

def compare_lbp_histograms(hist1, hist2):
    # Ensure histograms are of type np.float32
    hist1 = hist1.astype(np.float32)
    hist2 = hist2.astype(np.float32)

    # Compute Chi-Squared distance
    distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    return distance
if __name__ == "__main__":
    # Load images in grayscale mode
    image1 = cv2.imread("fake_phonepe.jpg", cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread("org_phonepe.jpg", cv2.IMREAD_GRAYSCALE)

    # Compute LBP histograms
    hist1 = compute_lbp_histogram(image1)
    hist2 = compute_lbp_histogram(image2)

    # Compare histograms
    distance = compare_lbp_histograms(hist1, hist2)
    print(f"Chi-Squared Distance: {distance}")

    # Based on the distance value, determine similarity
    if distance < 0.1:  # Set a threshold based on your requirements
        print("The textures are similar.")
    else:
        print("The textures are different.")


# SHAPE ANALYSIS
        import cv2
def compare_shapes(image1_path, image2_path, threshold=0.1):  # You can adjust the threshold value
    # Read images in grayscale mode
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Apply Canny edge detection
    edges1 = cv2.Canny(img1, 100, 200)
    edges2 = cv2.Canny(img2, 100, 200)

    # Find contours in the edge images
    contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate Hu Moments for the first contour in each image
    if len(contours1) == 0 or len(contours2) == 0:
        return "No shapes detected."

    moments1 = cv2.moments(contours1[0])
    moments2 = cv2.moments(contours2[0])

    hu_moments1 = cv2.HuMoments(moments1).flatten()
    hu_moments2 = cv2.HuMoments(moments2).flatten()

    # Compare Hu Moments using matchShapes
    similarity_score = cv2.matchShapes(hu_moments1, hu_moments2, cv2.CONTOURS_MATCH_I2, 0.0)

    # Return result based on similarity score
    if similarity_score < 0.1:
        return "Shapes are similar."
    else:
        return "Shapes are different."

if __name__ == "__main__":
    image1_path = "fake_phonepe.jpg"
    image2_path = "org_phonepe.jpg"

    result = compare_shapes(image1_path, image2_path)
    print(result)


 #COLOR ANALYSIS
import cv2
import numpy as np

def compare_color_histograms(image1_path, image2_path):
    # Load images in BGR mode
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Convert images to RGB (OpenCV loads images in BGR format)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Compute histograms for each channel (R, G, B)
    hist1_r = cv2.calcHist([img1_rgb], [0], None, [256], [0, 256])
    hist1_g = cv2.calcHist([img1_rgb], [1], None, [256], [0, 256])
    hist1_b = cv2.calcHist([img1_rgb], [2], None, [256], [0, 256])

    hist2_r = cv2.calcHist([img2_rgb], [0], None, [256], [0, 256])
    hist2_g = cv2.calcHist([img2_rgb], [1], None, [256], [0, 256])
    hist2_b = cv2.calcHist([img2_rgb], [2], None, [256], [0, 256])

    # Combine histograms
    hist1 = np.concatenate((hist1_r, hist1_g, hist1_b), axis=None)
    hist2 = np.concatenate((hist2_r, hist2_g, hist2_b), axis=None)

    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # Compute Chi-Squared distance
    chi_squared_distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

    return chi_squared_distance

if __name__ == "__main__":
    image1_path = "org_phonepe.jpg"
    image2_path = "fake_phonepe.jpg"

    distance = compare_color_histograms(image1_path, image2_path)
    print(f"Chi-Squared Distance: {distance}")

    # Based on the distance value, determine if images have similar color distributions
    if distance < 0.1:  # Adjust this threshold based on your requirements
        print("The color distributions are similar.")
    else:
        print("The color distributions are different.")

# FREQUENCY DOMAIN ANALYSIS
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_frequency_domains(img1, img2):
    # Convert images to grayscale
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute Fourier Transforms
    f_img1 = np.fft.fft2(gray_img1)
    f_img2 = np.fft.fft2(gray_img2)

    # Shift the zero frequency component to the center
    fshift_img1 = np.fft.fftshift(f_img1)
    fshift_img2 = np.fft.fftshift(f_img2)

    # Compute magnitude spectrum
    magnitude_spectrum1 = 20 * np.log(np.abs(fshift_img1))
    magnitude_spectrum2 = 20 * np.log(np.abs(fshift_img2))

    # Ensure both spectra have the same shape (pad if necessary)
    min_shape = min(magnitude_spectrum1.shape[0], magnitude_spectrum2.shape[0])
    magnitude_spectrum1 = magnitude_spectrum1[:min_shape, :]
    magnitude_spectrum2 = magnitude_spectrum2[:min_shape, :]

    # Compute the difference between magnitude spectra
    difference = np.sum(np.abs(magnitude_spectrum1 - magnitude_spectrum2))

    return difference

if __name__ == "__main__":
    # Load two images
    img1 = cv2.imread("org_phonepe.jpg")
    img2 = cv2.imread("fake_phonepe.jpg")

    # Compare frequency domains
    difference = compare_frequency_domains(img1, img2)
    print(f"Difference in Frequency Domains: {difference}")


    # Decide if images are similar or different based on the difference value
    if difference < 1.0:  # Set a threshold value based on your requirements
        print("The images are similar.")
    else:
        print("The images are different.")




# JACCARD SIMILARITY AND INTERSECTION OVER UNION (IoU) 
import cv2
import numpy as np

def binary_threshold(image, threshold=128):
    """Convert an image to binary format using a threshold."""
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def compute_jaccard_similarity(image1, image2):
    """Compute Jaccard Similarity."""
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)
    return intersection.sum() / float(union.sum())

def resize_image(image, target_shape):
    """Resize an image to a target shape."""
    return cv2.resize(image, target_shape[::-1], interpolation=cv2.INTER_NEAREST)

if __name__ == "__main__":
    # Load images in grayscale mode
    image1 = cv2.imread("org_phonepe.jpg", cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread("fake_phonepe.jpg", cv2.IMREAD_GRAYSCALE)

    # Convert images to binary format
    binary1 = binary_threshold(image1)
    binary2 = binary_threshold(image2)

    # Check and resize images to ensure the same shape
    if binary1.shape != binary2.shape:
        target_shape = binary1.shape
        binary1 = resize_image(binary1, target_shape)
        binary2 = resize_image(binary2, target_shape)

    # Compute Jaccard Similarity
    jaccard_similarity = compute_jaccard_similarity(binary1, binary2)
    print(f"Jaccard Similarity: {jaccard_similarity}")

    # Determine if images are similar or different based on thresholds
    if jaccard_similarity < 0.9:  # Adjust the threshold as needed
        print("The images are ACCURATE.")
    else:
        print("The images are SUS.")


# BLUR AND SHARPNESS COMPARISON
import cv2
import numpy as np

def compute_blur_and_sharpness(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Compute Laplacian of the image
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    
    # Compute the variance and mean of the Laplacian
    variance = laplacian.var()
    mean = np.mean(laplacian)
    
    return variance, mean

def compare_images(image1_path, image2_path, threshold=500):
    # Compute blur and sharpness metrics for both images
    var1, mean1 = compute_blur_and_sharpness(image1_path)
    var2, mean2 = compute_blur_and_sharpness(image2_path)
    
    # Compare the metrics using a threshold
    blur_similarity = abs(var1 - var2) > threshold
    sharpness_similarity = abs(mean1 - mean2) > threshold
    
    return blur_similarity, sharpness_similarity

if __name__ == "__main__":
    image1_path = "fake_phonepe.jpg"
    image2_path = "org_phonepe.jpg"

    blur_sim, sharpness_sim = compare_images(image1_path, image2_path)
    
    if blur_sim and sharpness_sim:
        print("Both images have similar blur and sharpness characteristics.")
    else:
        print("The images have different blur and/or sharpness characteristics.")
