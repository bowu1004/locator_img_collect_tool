import cv2
import numpy as np

def calculate_bhatt_coefficient(hist1, hist2):
    # Calculate Bhattacharyya coefficient between two histograms
    coeff = np.sum(np.sqrt(hist1 * hist2))
    return coeff

def calculate_similarity(image1, image2):
    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate histograms for each image
    hist1 = cv2.calcHist([gray_image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray_image2], [0], None, [256], [0, 256])

    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # Calculate Bhattacharyya coefficient
    similarity = calculate_bhatt_coefficient(hist1, hist2)

    return similarity

# Read images
image1 = cv2.imread('dataset/train/coco_cola/00000.png')
image2 = cv2.imread('dataset/train/coco_cola/000016.png')

# Calculate similarity
similarity = calculate_similarity(image1, image2)

# Print similarity value
print('Similarity:', similarity)
