from flask import Flask, request, jsonify
import cv2
import numpy as np
import io
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import pytesseract
import re

DEBUG=False

app = Flask(__name__)


# Utility Functions
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
    if DEBUG:
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

def extract_boxes(image):
    # Load the image
    height, width = image.shape[:2]

# PROCESSING BOX1

    # Define the regions based on the layout
    box1 = image[0:(height//2), 0:(width*3//5)]  # Left side, top half
    box1 = box1[(height//8):height, (width//16):(width*17//30)]
    
    # Adjust contrast (alpha) and brightness (beta)
    box1 = cv2.convertScaleAbs(box1, alpha=0.75, beta=0)  # Increase contrast by setting alpha > 1
    # Morphological dilation to thicken text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # You can adjust the kernel size
    # dilating the image to get finer details
    box1 = cv2.dilate(box1, kernel, iterations=3)

# PROCESSING BOX2
    
    # Define the regions based on the layout
    box2 = image[0:height, ((width*3)//5):width]  # Right side, full height
    box2 = box2[((height*4)//34):((height*19)//20), (width//50):(width*7//20)]    
    
    # Adjust contrast (alpha) and brightness (beta)
    box2 = cv2.convertScaleAbs(box2, alpha=0.75, beta=0)  # Increase contrast by setting alpha > 1
    # Morphological dilation to thicken text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # You can adjust the kernel size
    # dilating the image to get finer details
    box2 = cv2.dilate(box2, kernel, iterations=3)    
    
    
# PROCESSING BOX3
    
    # Define the regions based on the layout
    box3 = image[height//2:height, 0:(width*3//5)]  # Left side, bottom half
    box3 = box3[(height//30):height, 0:width]

    # Adjust contrast (alpha) and brightness (beta)
    box3 = cv2.convertScaleAbs(box3, alpha=0.75, beta=0)  # Increase contrast by setting alpha > 1
    # Morphological dilation to thicken text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # You can adjust the kernel size
    # dilating the image to get finer details
    box3 = cv2.dilate(box3, kernel, iterations=3)  

# OUTPUTTING and image debugging

    # Save each section as a separate image - Unnecessary but used for debugging
    cv2.imwrite("box1.png", box1)
    cv2.imwrite("box2.png", box2)
    cv2.imwrite("box3.png", box3)

    # Return the images if further processing is needed
    return box1, box2, box3

def extract_text(image):
    # Open the image file
    pil_image = Image.fromarray(image)
    
    # Use Tesseract to extract text
    return pytesseract.image_to_string(pil_image)

# def extract_clues(clue_text):
#     # Regular expression pattern to match clue numbers and corresponding text
#     pattern = r"(\d+)\)\s*(.*?)(?=\s*\d+\)|$)"

#     # Find all matches using the regex
#     matches = re.findall(pattern, clue_text, re.DOTALL)

#     # Initialize an empty dictionary to store the clues
#     clues_dict = {}

#     # Loop through the matches and populate the dictionary
#     for match in matches:
#         clue_number = match[0]
#         clue_text = match[1].replace("\n", " ").strip()  # Replace newlines in clues with a space
#         clues_dict[clue_number] = clue_text

#     # Print the dictionary to see the result
#     return clues_dict

def extract_clues(clue_text):
    """
    Extracts clues from the given text while accounting for multiline values and proper key-value separation.

    Args:
        clue_text (str): Raw text containing clue numbers and descriptions.

    Returns:
        dict: A dictionary with clue numbers as keys and clue descriptions as values.
    """
    # Split the input text into individual lines for processing
    lines = clue_text.splitlines()

    # Initialize the dictionary to store the extracted clues
    clues_dict = {}
    current_key = None
    current_clue = None

    # Pattern to match clue number and text (handles special characters and multiline clues)
    pattern = re.compile(r'([^\d]*\d+[^\)]*)\)\s*(.*)')

    for line in lines:
        line = line.strip()

        # Check if the line matches the clue number pattern
        match = pattern.match(line)

        if match:
            # If we match a new clue, add the previous clue (if exists)
            if current_key:
                clues_dict[current_key] = current_clue.strip()

            # Extract new key and clue
            current_key = match.group(1).strip()  # This may include special characters like '§0'
            current_clue = match.group(2).strip()

        elif current_key:
            # If there is no key but we have a previous key, it's a continuation of the previous clue
            current_clue += " " + line.strip()

    # Add the last clue to the dictionary
    if current_key:
        clues_dict[current_key] = current_clue.strip()

    return clues_dict

"""
Data cleansing is being done after creating dictionaries of the clues
This is so cleansing of numbers for the clues can be done independently
from cleansing for the clues themselves 
"""  

def data_clean_dict(clues_dict):
    """ 
    Cleans up the clue numbers and values in a dictionary using one loop.
    Args: clues_dict (dict): Dictionary with clue numbers as keys and clues as values.
    Returns: dict: A new dictionary with corrected clue numbers as keys 
    and clues without quotation marks as values 
    """
    # Defined mapping for known incorrect symbols to numbers
    symbol_to_number = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'l': '1', '|': '1', '/': '1',
    'H': '11',    
    'Z': '2',
    'E': '33',
    'A': '4', 'h': '4',        
    'S': '5', '§': '5',
    'G': '6', 'b': '6',
    'T': '7',    
    'B': '8', '%': '8',
    'g': '9', 'q': '9'
    }
    
    cleaned_dict = {}
    
    for key, value in clues_dict.items():
        # Attempt to correct the key if it's not a digit
        if not key.isdigit():
            corrected_key = "".join(symbol_to_number.get(char, char) for char in key)
        else:
            corrected_key = key
        
        # Remove unwanted quotation marks from the value
        cleaned_value = value.replace('"', '').replace('“', '').replace('”', '')
        
        # Add the cleaned pair to the new dictionary
        cleaned_dict[corrected_key] = cleaned_value
    
    return cleaned_dict

# Function to get the 2D matrix from the image
def extract_2d_matrix(image):
    """Extract the 2D matrix for the image."""
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Detect edges
    edges = detect_edges(processed_image)

    # Find contours and the document corners
    corners = find_contours(edges)
    if corners is None:
        raise ValueError("No document-like contour found.")

    # Get the perspective transformation matrix
    matrix = get_transformation_matrix(corners)

    return matrix


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

        # Break image into boxes box1_across, box2_down, box3_matrix
        box1_across, box2_down, box3_matrix = extract_boxes(transformed_image)
        
        # Extract text from box1_across and box2_down for the across and down clues
        box1_text = extract_text(box1_across)
        box2_text = extract_text(box2_down)
            
        box1_clue_dict = extract_clues(box1_text)
        box2_clue_dict = extract_clues(box2_text)
        
        box1_clue_dict = data_clean_dict(box1_clue_dict)
        box2_clue_dict = data_clean_dict(box2_clue_dict)
        
        # Calls to get the 2d matrix
        box3_2d_matrix = extract_2d_matrix(box3_matrix)
        
                
        box1_across_hex = image_to_hex(box1_across)
        box2_down_hex = image_to_hex(box2_down)
        box3_matrix_hex = image_to_hex(box3_matrix)

        return jsonify({
            "original_image": original_image_hex,
            "transformed_image": transformed_image_hex,
            "corners": corners.tolist(),  # Return corner values for further processing if needed
            "box1_across": box1_across_hex,
            "box2_down": box2_down_hex,
            "box3_matrix": box3_matrix_hex,
        })
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
