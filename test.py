from flask import Flask, request, jsonify
import cv2
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt


app = Flask(__name__)

def load_image(file):
    """Load an image from the request file into a NumPy array."""
    in_memory_file = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to load image. Check the file format.")
    return image

# Functions to convert to and from grayscale and rgb color
def convert_to_grayscale(image):
    """Convert the image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_to_rgb(image):
    """Convert the image from BGR to RGB for display purposes."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Helper Function to save image or display image at different stages for debugging
def save_or_show_image(image, title="Image", color_map=None):
    """Display or save the image at different stages for debugging."""
    plt.figure()
    plt.imshow(image, cmap=color_map)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Function to convert Numpy image array to dex string for JSON response
def image_to_hex(image):
    """Convert a NumPy image array to a hexadecimal string for JSON response."""
    pil_image = Image.fromarray(image)
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr.seek(0)
    return byte_arr.getvalue().hex()

# Preprocessing image in order (grayscale, blur, bilateral filtering, binary threshold)
def preprocess_image(image):
    """Preprocess the image to enhance edge detection for document."""
    # Convert to grayscale
    grayscale_image = convert_to_grayscale(image)
    
    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    
    # Apply Bilateral Filtering for further noise reduction while keeping edges sharp
    filtered_image = cv2.bilateralFilter(blurred_image, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Apply binary threshold to get a black and white effect
    _, black_and_white_image = cv2.threshold(filtered_image, 128, 255, cv2.THRESH_BINARY)
    
    return black_and_white_image

# Detects edges using Canny
def detect_edges(image):
    """Detect edges using Canny edge detection."""
    return cv2.Canny(image, 50, 150)

# Takes the edges returned from detect_edges to get the specific document 4 corners
def find_contours(edges):
    """Find contours and select the one with 4 corners that represents the document."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:  # Found a quadrilateral
            print(f"Found a quadrilateral (paper) contour: {approx}")
            return approx.reshape(4, 2)  # Return the 4 corner points

    return None  # No suitable contour found

# Takes corner_points returned from find_contours to flatten the document
def get_transformation_matrix(corner_points):
    """Get transformation matrix to flatten the document."""
    dst_points = np.float32([[2480, 0], [0, 0], [0, 3508], [2480, 3508]])  # A4 paper size
    matrix = cv2.getPerspectiveTransform(corner_points, dst_points)
    return matrix

# Transforms the perspective of the image to specified size.
def apply_perspective_transform(image, matrix, size=(2480, 3508)):
    """Apply perspective transformation to flatten the document."""
    return cv2.warpPerspective(image, matrix, size)

# # Detects corners using Harris
# def detect_corners(grayscale_image, block_size=2, ksize=3, k=0.04, threshold_factor=0.01):
#     """Apply Harris Corner Detection and highlight corners on the original image."""
#     grayscale_float = np.float32(grayscale_image)
#     corners = cv2.cornerHarris(grayscale_float, blockSize=block_size, ksize=ksize, k=k)
#     corners = cv2.dilate(corners, None)  # Dilate to enhance corner points
#     threshold = threshold_factor * corners.max()
#     return corners, threshold

def apply_perspective_transform(image, matrix, size=(2480, 3508)):
    """Apply perspective transformation to flatten the grid."""
    return cv2.warpPerspective(image, matrix, size)

@app.route('/', methods=['POST'])
def upload_image():
    # Check if the request contains a file
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    
    file = request.files['image']
    
    # Check if an image file was uploaded
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    try:
        
         # Load and process the image
        image = load_image(file)
        
        preprocessed_image = preprocess_image(image)
        
        # Detect edges and find document contour
        edges = detect_edges(preprocessed_image)
        
        corners = find_contours(edges)
        
        if corners is None:
            return jsonify({"error": "Document edges not detected"}), 400
        
        if corners is not None:
            contour_image = image.copy()
            cv2.drawContours(contour_image, [corners], -1, (0, 255, 0), 3)
            save_or_show_image(contour_image, title="Detected Document Contour Edges")
        else:
            print("No contour with 4 corners found.")

        # Perspective transformation (if corners were found)
        if corners is not None:
            matrix = get_transformation_matrix(np.float32(corners))
            transformed_image = apply_perspective_transform(image, matrix)
            save_or_show_image(transformed_image, title="Transformed (Flattened) Document")
        else:
            print("Cannot apply perspective transform as corners were not detected.")
        
        # Convert images to RGB for response
        original_rgb = convert_to_rgb(image)
        transformed_rgb = convert_to_rgb(transformed_image)

        # Convert images to hex strings
        original_image_hex = image_to_hex(original_rgb)
        transformed_image_hex = image_to_hex(transformed_rgb)

        return jsonify({
            "original_image": original_image_hex,
            "transformed_image": transformed_image_hex,
            "corners": corners.tolist()  # Return corner values for further processing if needed
        })
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)