import cv2
import numpy as np
import pytesseract
import re

# * Main Function to extract the boxes Across, Down, and Matrix *
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
    # M - cutting down image to only the matrix
    box3 = image[((height//2)-10)+(height//30):(height-(height//25)), ((width//22)):(width*3//5)-7]
    # cv2.imwrite("box3_check.png", box3)

    # OUTPUTTING and image debugging

    # Save each section as a separate image - Unnecessary but used for debugging
    # cv2.imwrite("box1.png", box1)
    # cv2.imwrite("box2.png", box2)
    # cv2.imwrite("box3.png", box3)

    # Return the images if further processing is needed
    return box1, box2, box3

def extract_text(image):
    """
    Use Tesseract to extract text directly from a NumPy image array using OpenCV.
    """
    # Ensure the image is in grayscale (required for better OCR performance)
    if len(image.shape) == 3:  # If the image is not already grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract to extract text from the image
    return pytesseract.image_to_string(image)


# Function to use RegEX to extract individual clues from text string of clue information
# *** Note: Uses ')' as the delimiter. * could be an issue if ocr picks up a 0, O, or o as ().
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

# Function to data cleanse the key,values of a clue dictionary from extract clues function
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

# Function to get the 2D matrix from the crossword image (box3)
def process_box3(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Debug original image properties
    # print("Original image shape:", image.shape, "dtype:", image.dtype)
    
    image = cv2.GaussianBlur(image, (7, 7), 0)
    # cv2.imwrite("debug_blur.png", image)
    
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, sharpening_kernel)
    # cv2.imwrite("debug_sharpened.png", image)

    mask = cv2.inRange(image, 0, 180)
    image[mask == 255] = 0
    # cv2.imwrite("debug_blacken_low_range.png", image)

    near_white_mask = cv2.inRange(image, 180, 255)
    image[near_white_mask == 255] = 255
    # cv2.imwrite("debug_whiten_high_range.png", image)

    # cv2.imwrite("process_debug.png", image)
    return image

# Helper function detects if there are pixels in the top right corner of the crossword individual boxes
# indicating there is a number in that box
def process_pixel(img, threshold=0.3):
    """
    Detects whether a cell contains a number based on the concentration of black pixels
    in the top-left region of the cell.

    Parameters:
        img (numpy.ndarray): The input image of the cell (grayscale).
        threshold (float): The percentage of black pixels required to classify the cell as containing a number.

    Returns:
        bool: True if a number is detected, False otherwise.
    """
    # Define the top-left region as a fraction of the cell's dimensions
    height, width = img.shape
    region_h, region_w = height // 2, width // 2  # Top-left quarter of the cell

    # Crop the top-left region
    top_left_region = img[0:region_h, 0:region_w]

    # Count the number of black pixels in the region
    black_pixel_count = np.sum(top_left_region < 128)  # Pixels with intensity < 128 are considered black

    # Calculate the percentage of black pixels
    total_pixels = region_h * region_w
    black_pixel_percentage = black_pixel_count / total_pixels

    # Check if the percentage exceeds the threshold
    return black_pixel_percentage > threshold

# Function that splices crossword into individual boxes and calls box_corner_number_detection
# on each box and constructs crossword matrix with numbers, '_' for white, and '#' for black.
def crossword_extract(image):
    # Process the image (ensure you define the 'process' function for preprocessing)

    # cv2.imwrite("process_box3_before.png", image)
    image = process_box3(image)  
    # cv2.imwrite("cross.png", image) 
    

    h = image.shape[0] // 15  # height of each cropped section
    w = image.shape[1] // 15  # width of each cropped section

    grid = []
    count = 1
    # Loop through each row and column for cropping
    for i in range(15):
        row = []
        for j in range(15):
            # Define the cropping coordinates
            y_start = i * h # initial start is 0 times the height
            y_end = (i + 1) * h # initital end is 1 times the height 
            x_start = j * w 
            x_end = (j + 1) * w

            cropped_image = image[y_start:y_end, x_start:x_end]
            # cv2.imwrite("crop.png", cropped_image)
            if count < 10:
                text = process_pixel(cropped_image, .23)
            else:
                text = process_pixel(cropped_image, .3)            

            # Check if the cropped section is mostly white or black
            avg_pixel_value = np.mean(cropped_image)
            
            # If the average pixel value is greater than 127, classify it as white
            if avg_pixel_value > 127:
                if text:  # If text is found
                    row.append(str(count))
                    count += 1
                else:
                    row.append("_")  # Append "white" if no text found
            else:
                row.append("#")

        # Add the row to the grid
        grid.append(row)

    return grid