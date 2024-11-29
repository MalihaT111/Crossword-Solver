from flask import Flask, request, jsonify
import cv2
import numpy as np
import io
import matplotlib.pyplot as plt
import pytesseract
import re
import requests
import urllib
from bs4 import BeautifulSoup

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

# Function to convert to grayscale
def convert_to_grayscale(image):
    """Convert the image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to convert to rgb color
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
    """Convert a NumPy image array to a hexadecimal string for JSON response using OpenCV."""
    # Ensure image is in the correct format (BGR to RGB if needed)
    if len(image.shape) == 3:  # Check if the image is a color image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image  # Grayscale images don't need conversion
    
    # Encode image as PNG
    success, buffer = cv2.imencode('.png', image_rgb)
    if not success:
        raise ValueError("Failed to encode image to PNG.")
    
    # Convert the buffer to hexadecimal
    return buffer.tobytes().hex()


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
    return cv2.Canny(image, 150, 150)

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
    
    # dst_points = np.float32([[0, 0], [0, 3508], [2480, 3508], [2480, 0]]) # crosswordZ transformation 
    
    matrix = cv2.getPerspectiveTransform(corner_points, dst_points)
    return matrix

# Transforms the perspective of the image to specified size.
def apply_perspective_transform(image, matrix, size=(2480, 3508)):
    """Apply perspective transformation to flatten the document."""
    return cv2.warpPerspective(image, matrix, size)

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
    cv2.imwrite("box1.png", box1)
    cv2.imwrite("box2.png", box2)
    cv2.imwrite("box3.png", box3)

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

# Function to generate the answer matrix based on the crossword matrix - # for black, _ for white
def generate_answer_matrix(crossword_matrix):
    answer_matrix = []

    for row in crossword_matrix:
        answer_row = []
        for cell in row:
            if cell == '#':
                answer_row.append('#')  # Keep black cells as 'black'
            elif cell.isdigit():  # If it's a number (clue starting point)
                answer_row.append('_')  # Replace numbers with None (or empty string)
            else:
                answer_row.append('_')  # For white cells
        answer_matrix.append(answer_row)

    return answer_matrix

# Function to get the pattern of clue, pattern, and length for individual website post request
def generate_pattern(crossword_matrix, answer_matrix, clue_numbers):
    pattern = []
    
    for clue_number in clue_numbers:
        # Find the cells that correspond to this clue
        clue_cells = get_clue_cells(crossword_matrix, clue_number)
        clue_pattern = ""
        
        for cell in clue_cells:
            row, col = cell
            if answer_matrix[row][col] is not None:
                clue_pattern += answer_matrix[row][col]  # If we already have an answer, use it
            else:
                clue_pattern += "?"  # Use "?" for unresolved cells
        
        pattern.append(clue_pattern)
    
    return pattern

# Function to get a dictionary of clue numbers to an array of clue possible answers
def generate_possible_clue_answers(clue_dict, crossword, answer_matrix, direction):
    """
    Generate a dictionary of possible answers for each clue.

    Parameters:
        clue_dict (dict): Dictionary of clues (clue_number: clue_text).
        crossword (list[list[str]]): The crossword grid.
        answer_matrix (list[list[str]]): The answer matrix.
        direction (str): Direction of clues ("across" or "down").

    Returns:
        dict: A dictionary with clue numbers as keys and lists of possible answers as values.
    """
    possible_clue_answers = {}

    for clue_number, clue_text in clue_dict.items():
        clue_number = int(clue_number)  # Ensure the clue number is an integer
        try:
            # Determine the pattern based on the direction
            if direction == "across":
                pattern = find_pattern_across(crossword, answer_matrix, clue_number)
            elif direction == "down":
                pattern = find_pattern_down(crossword, answer_matrix, clue_number)
            else:
                raise ValueError("Invalid direction. Use 'across' or 'down'.")
            
            # Fetch possible answers for the clue
            possible_answers = fetch_crossword_answers(clue_text, pattern, len(pattern))
            
            # Store the results in the dictionary
            possible_clue_answers[clue_number] = possible_answers

        except ValueError as e:
            print(f"Skipping clue {clue_number}: {e}")
        except Exception as e:
            print(f"Error processing clue {clue_number}: {e}")

    return possible_clue_answers

# Helper Function to find the starting possition of the clue in the crossword puzzle
def find_starting_location(crossword, clue_number):
    """
    Locate the starting row and column of the given clue number.
    
    Parameters:
        crossword (list[list[str]]): The crossword grid.
        clue_number (int): The clue number to locate.
    
    Returns:
        tuple[int, int]: The (row, column) of the starting clue.
    """
    for row_idx, row in enumerate(crossword):
        for col_idx, cell in enumerate(row):
            if cell == str(clue_number):
                # Found the start location of the clue
                return row_idx, col_idx
    raise ValueError(f"Clue number {clue_number} not found in the crossword.")

# Helper Function to find the request string pattern p if the clue is in across clue
def find_pattern_across(crossword, answer_matrix, clue_number):
    """
    Finds the pattern for a given clue number across in the crossword.
    
    Parameters:
        crossword (list[list[str]]): The crossword grid with clue numbers and '#' for black spaces.
        answer_matrix (list[list[str]]): The corresponding answer grid with '_' for empty spaces.
        clue_number (int): The clue number to find the pattern for.
    
    Returns:
        str: The pattern as a string (e.g., "????" or "A??T").
    """
    # Locate the clue number in the crossword grid
    start_row, start_col = find_starting_location(crossword, clue_number)

    # Build the pattern by traversing across the row
    pattern = []
    for col in range(start_col, len(crossword[start_row])):
        cell = crossword[start_row][col]
        if cell == "#":  # Stop at a black space
            break
        # Use the corresponding cell in the answer matrix
        answer_cell = answer_matrix[start_row][col]
        if answer_cell == "_":  # Empty space becomes "?"
            pattern.append("?")
        else:  # Filled letters are preserved
            pattern.append(answer_cell)
    
    return "".join(pattern)

# Helper Function to find the request string pattern p if the clue is a down clue
def find_pattern_down(crossword, answer_matrix, clue_number):
    """
    Finds the pattern for a given clue number down in the crossword.
    
    Parameters:
        crossword (list[list[str]]): The crossword grid with clue numbers and '#' for black spaces.
        answer_matrix (list[list[str]]): The corresponding answer grid with '_' for empty spaces.
        clue_number (int): The clue number to find the pattern for.
    
    Returns:
        str: The pattern as a string (e.g., "????" or "A??T").
    """
    # Locate the clue number in the crossword grid
    start_row, start_col = find_starting_location(crossword, clue_number)

    # Build the pattern by traversing downward the column
    pattern = []
    for row in range(start_row, len(crossword)):
        cell = crossword[row][start_col]
        if cell == "#":  # Stop at a black space
            break
        # Use the corresponding cell in the answer matrix
        answer_cell = answer_matrix[row][start_col]
        if answer_cell == "_":  # Empty space becomes "?"
            pattern.append("?")
        else:  # Filled letters are preserved
            pattern.append(answer_cell)
    
    return "".join(pattern)

# Function to fetch an array of answers based on a clue, pattern string, and length.
# Note for URL requests: "%3F" is a reserved URI character representing "?"
def fetch_crossword_answers(clue: str, pattern: str = None, length: int = None):
    """
    Fetch possible crossword answers for a given clue, with optional pattern and length.
    Tries more general queries if specific queries return no results.
    
    Parameters:
        clue (str): The crossword clue text.
        pattern (str, optional): The clue pattern (e.g., '????'). Default is None.
        length (int, optional): The expected length of the answer. Default is None.
    
    Returns:
        list: A list of possible answers.
    """
    base_url = "https://www.dictionary.com/e/crosswordsolver/"
    url = base_url + f"{urllib.parse.quote(clue)}/"  # Encode clue in the URL

    # Define fallback query scenarios
    queries = []
    if pattern and length:
        queries.append({"p": pattern, "l": length})  # Specific: pattern and length
    if pattern:
        queries.append({"p": pattern})              # Less specific: only pattern
    queries.append({})                              # Least specific: no parameters

    for params in queries:
        try:
            # Build the request URL with the current parameters
            request_url = url + f"?{urllib.parse.urlencode(params)}" if params else url
            response = requests.get(request_url)
            response.raise_for_status()  # Raise error if request fails

            # Parse the HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            rows = soup.find_all('div', class_='solver-table__row')

            # Extract answers from the rows
            answers = []
            for row in rows:
                answer_cell = row.find('div', attrs={'data-cy': 'result'})
                if answer_cell:
                    answer = answer_cell.text.strip()
                    answers.append(answer)
            
            if answers:
                return answers  # Return results as soon as we find any
            
        except requests.RequestException as e:
            print(f"Error fetching crossword answers for params {params}: {e}")
    
    return []  # Return an empty list if all queries fail

def get_next_clue(across_clues, down_clues):
    """
    Selects the next clue to solve based on the fewest options available.

    Parameters:
        across_clues (dict): Dictionary of across clues with possible answers.
        down_clues (dict): Dictionary of down clues with possible answers.

    Returns:
        clue_number (int): The number of the next clue to solve.
        direction (str): "across" or "down" to indicate clue direction.
        possible_answers (list): List of possible answers for that clue.
    """
    # Combine across and down clues into a list of all clues
    all_clues = [(k, 'across', v) for k, v in across_clues.items()] + \
                [(k, 'down', v) for k, v in down_clues.items()]
    
    # Sort clues by the number of possible answers (ascending)
    all_clues.sort(key=lambda x: len(x[2]))
    
    # Return the clue with the fewest possible answers
    return all_clues[0]  # This returns (clue_number, direction, possible_answers)

def is_valid_answer(answer_matrix, clue_number, answer, direction, crossword):
    """
    Checks if placing an answer in the grid is valid.
    
    Parameters:
        answer_matrix (list[list[str]]): The current answer grid.
        clue_number (int): The clue number to which the answer corresponds.
        answer (str): The candidate answer to place in the grid.
        direction (str): "across" or "down", the direction of the clue.
        crossword (list[list[str]]): The crossword structure.

    Returns:
        bool: True if the answer is valid, False otherwise.
    """
    # Get the starting position for the clue
    start_row, start_col = find_clue_position(crossword, clue_number, direction)
    
    if direction == 'across':
        # Check horizontal validity: Make sure answer fits and does not conflict with existing answers
        for i in range(len(answer)):
            if crossword[start_row][start_col + i] != '#' and answer_matrix[start_row][start_col + i] not in ('_', answer[i]):
                return False
    elif direction == 'down':
        # Check vertical validity: Make sure answer fits and does not conflict with existing answers
        for i in range(len(answer)):
            if crossword[start_row + i][start_col] != '#' and answer_matrix[start_row + i][start_col] not in ('_', answer[i]):
                return False
    
    return True

def find_clue_position(crossword, clue_number, direction):
    """
    Finds the starting position (row, column) for a given clue in the crossword.
    
    Parameters:
        crossword (list[list[str]]): The crossword structure.
        clue_number (int): The clue number to locate.
        direction (str): "across" or "down", the direction of the clue.

    Returns:
        (int, int): The row and column of the clue's starting position.
    """
    for row in range(len(crossword)):
        for col in range(len(crossword[row])):
            if crossword[row][col] == str(clue_number):
                # Found the clue number
                if direction == 'across' and col < len(crossword[row]) - 1 and crossword[row][col + 1] != '#':
                    return row, col
                elif direction == 'down' and row < len(crossword) - 1 and crossword[row + 1][col] != '#':
                    return row, col
    return None  # In case the clue is not found (shouldn't happen if the crossword is correct)

def place_answer(answer_matrix, clue_number, answer, direction):
    """
    Places the given answer into the grid at the correct positions.
    
    Parameters:
        answer_matrix (list[list[str]]): The current answer grid.
        clue_number (int): The clue number to which the answer corresponds.
        answer (str): The answer to place in the grid.
        direction (str): "across" or "down", the direction of the clue.
    """
    start_row, start_col = find_clue_position(crossword, clue_number, direction)
    
    if direction == 'across':
        for i in range(len(answer)):
            answer_matrix[start_row][start_col + i] = answer[i]
    elif direction == 'down':
        for i in range(len(answer)):
            answer_matrix[start_row + i][start_col] = answer[i]

def remove_answer(answer_matrix, clue_number, answer, direction):
    """
    Removes the given answer from the grid (backtracking).
    
    Parameters:
        answer_matrix (list[list[str]]): The current answer grid.
        clue_number (int): The clue number to which the answer corresponds.
        answer (str): The answer to remove from the grid.
        direction (str): "across" or "down", the direction of the clue.
    """
    start_row, start_col = find_clue_position(crossword, clue_number, direction)
    
    if direction == 'across':
        for i in range(len(answer)):
            answer_matrix[start_row][start_col + i] = '_'
    elif direction == 'down':
        for i in range(len(answer)):
            answer_matrix[start_row + i][start_col] = '_'

# functions to solve crossword
def positions(grid, clue_number, direction):
    """
    Determine the length of the word for a clue based on its position and direction.

    Args:
        grid (list[list[str]]): The crossword grid.
        clue_number (int): The clue number.
        direction (str): "across" or "down".

    Returns:
        int: The length of the word.
        list: Ordered list of positions as (row, col).
        tuple: Starting position as (row, col).
    """
    start_pos = ()
    positions = []

    # Find the starting position of the clue
    start_row, start_col = None, None
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if grid[row][col] == str(clue_number):
                start_row, start_col = row, col
                start_pos = (row, col)
                break
        if start_row is not None:
            break

    if start_row is None or start_col is None:
        raise ValueError(f"Clue {clue_number} not found in the grid.")

    # Collect positions based on the direction
    if direction == "across":
        col = start_col
        while col < len(grid[0]) and grid[start_row][col] != '#':
            positions.append((start_row, col))
            col += 1
    elif direction == "down":
        row = start_row
        while row < len(grid) and grid[row][start_col] != '#':
            positions.append((row, start_col))
            row += 1
    else:
        raise ValueError(f"Invalid direction: {direction}")

    word_length = len(positions)
    return word_length, positions, start_pos

# Function to sort clues by least constrained variable
def sort_clues_by_constraints(clues):
    """
    Sort clues based on the number of possible words (least constrained first).

    Args:
        clues (dict): Dictionary of clues.

    Returns:
        list: Sorted list of clues (clue_number, details).
    """
    return sorted(clues.items(), key=lambda item: -len(item[1][0]))

# now i have a function to go through the dictionary and append the proper things 
def update_clues_with_positions(grid, clues, direction):
    """
    Updates the clues dictionary with word length, start position, and positions set.

    Args:
        grid (list[list[str]]): The crossword grid.
        clues (dict): Dictionary of clues where keys are clue numbers, and values are possible words.
        direction (str): "across" or "down".

    Returns:
        dict: Updated dictionary with each clue number having (possible words array, word length, start pos, positions set).
    """
    updated_clues = {}
    for clue_number, words in clues.items():
        # Get the positions information for the clue
        word_length, positions_set, start_pos = positions(grid, clue_number, direction)
        # Update the clue value with the new information
        updated_clues[clue_number] = (words, word_length, start_pos, positions_set)
    return updated_clues

def solve_crossword(grid, across_clues, down_clues):
    """
    Solve the crossword puzzle with separate handling for across and down clues.
    
    Args:
        grid (list[list[str]]): The crossword grid.
        across_clues (dict): Updated across clues with possible words, word length, start pos, positions set.
        down_clues (dict): Updated down clues with possible words, word length, start pos, positions set.
    
    Returns:
        list[list[str]]: Partially or fully solved crossword grid.
    """


    def word_fits(grid, word, positions):
        """Check if a word fits in the current grid state at specified positions."""
        for (row, col), char in zip(positions, word):
            if grid[row][col] not in (char, "."):  # "." denotes an empty cell
                return False
        return True

    def place_word(grid, word, positions, clue_number):
        """Place a word in the grid and record ownership."""
        print(f"Placing word '{word}' for clue {clue_number} at positions {positions}")
        for (row, col), char in zip(positions, word):
            grid[row][col] = char
        print_grid(grid)

    def can_place_word(grid, word, positions):
        """
        Check if a word can be placed in the grid without causing conflicts.

        Args:
            grid (list[list[str]]): The crossword grid.
            word (str): The word to check.
            positions (list[tuple]): List of (row, col) positions for the word.

        Returns:
            bool: True if the word can be placed without conflicts, False otherwise.
        """
        for (row, col), letter in zip(positions, word):
            # Conflict if the cell is not empty and doesn't match the letter
            if grid[row][col] != '_' and grid[row][col] != letter:
                return False
        return True

    def remove_word(grid, word, positions):
        """
        Remove a word from the grid safely.

        Args:
            grid (list[list[str]]): The crossword grid.
            word (str): The word to remove.
            positions (list): List of positions (row, col) where the word was placed.
        """
        print(f"Removing word '{word}' from positions {positions}")
        for (row, col), char in zip(positions, word):
            # Reset only if the current letter matches the word being removed
            if grid[row][col] == char:
                grid[row][col] = "."
        print_grid(grid)  # Print the grid after removing the word

    def print_grid(grid):
        """Print the crossword grid."""
        print("\n".join(" ".join(cell if cell != "." else "_" for cell in row) for row in grid))
        print("\n" + "-" * 40 + "\n")
        
    def backtrack(clues, grid):
        """
        Recursive backtracking solver for clues using least constrained variables first.
        """
        if not clues:
            print("All clues solved!")
            print_grid(grid)  # Final solved grid
            return True  # No more clues to solve

        # Sort clues by LCV (most possible words first)
        sorted_clues = sort_clues_by_constraints(clues)

        for clue_number, (possible_words, word_length, start_pos, positions_set) in sorted_clues:
            print(f"Trying to solve clue {clue_number}...")

            for word in possible_words:
                if len(word) != word_length:
                    continue  # Skip words with incorrect lengths

                # Check if the word fits in the grid
                if word_fits(grid, word, positions_set):
                    place_word(grid, word, positions_set, clue_number)

                    # Remove the clue and recurse
                    del clues[clue_number]
                    if backtrack(clues, grid):
                        return True

                    # If it didn't work, backtrack (remove the word)
                    print(f"Backtracking: Removing word '{word}' for clue {clue_number}")
                    clues[clue_number] = (possible_words, word_length, start_pos, positions_set)
                    remove_word(grid, word, positions_set)

        print(f"Could not solve clue {clue_number}. Backtracking...")
        return False  # No valid words fit



    # Initialize the grid with empty cells
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] != "#":  # "#" represents blocked cells
                grid[row][col] = "."

    # Solve across clues first, then down clues
    backtrack(across_clues.copy(), grid)
    backtrack(down_clues.copy(), grid)

    return grid





grid2 = [
['_', '_', '_', '_', '#', '_', '_', '_', '_', '_', '#', '_', '_', '_', '_'],
['_', '_', '_', '_', '#', '_', '_', '_', '_', '_', '#', '_', '_', '_', '_'],
['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
['_', '_', '_', '_', '_', '_', '_', '_', '_', '#', '_', '_', '_', '#', '#'],
['_', '_', '_', '#', '_', '_', '_', '#', '#', '_', '_', '_', '_', '_', '_'],
['_', '_', '_', '_', '#', '#', '_', '_', '_', '_', '_', '#', '_', '_', '_'],
['#', '#', '#', '_', '_', '_', '#', '_', '_', '_', '_', '_', '_', '_', '_'],
['#', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '#'],
['_', '_', '_', '_', '_', '_', '_', '_', '#', '_', '_', '_', '#', '#', '#'],
['_', '_', '_', '#', '_', '_', '_', '_', '_', '#', '#', '_', '_', '_', '_'],
['_', '_', '_', '_', '_', '_', '#', '#', '_', '_', '_', '#', '_', '_', '_'],
['#', '#', '_', '_', '_', '#', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
['_', '_', '_', '_', '#', '_', '_', '_', '_', '_', '#', '_', '_', '_', '_'],
['_', '_', '_', '_', '#', '_', '_', '_', '_', '_', '#', '_', '_', '_', '_']
]
grid = [
    ['1', '2', '3', '4', '#', '5', '6', '7', '8', '9', '#', '10', '11', '12', '13'],
    ['14', '_', '_', '_', '#', '15', '_', '_', '_', '_', '#', '16', '_', '_', '_'],
    ['17', '_', '_', '_', '18', '_', '_', '_', '_', '_', '19', '_', '_', '_', '_'],
    ['20', '_', '_', '_', '_', '_', '_', '_', '_', '#', '21', '_', '_', '#', '#'],
    ['22', '_', '_', '#', '23', '_', '_', '#', '#', '24', '_', '_', '_', '25', '26'],
    ['27', '_', '_', '28', '#', '#', '29', '30', '31', '_', '_', '#', '32', '_', '_'],
    ['#', '#', '#', '33', '34', '35', '#', '36', '_', '_', '_', '37', '_', '_', '_'],
    ['#', '38', '39', '_', '_', '_', '40', '_', '_', '_', '_', '_', '_', '_', '#'],
    ['41', '_', '_', '_', '_', '_', '_', '_', '#', '42', '_', '_', '#', '#', '#'],
    ['43', '_', '_', '#', '44', '_', '_', '_', '45', '#', '#', '46', '47', '48', '49'],
    ['50', '_', '_', '51', '_', '_', '#', '#', '52', '53', '54', '#', '55', '_', '_'],
    ['#', '#', '56', '_', '_', '#', '57', '58', '_', '_', '_', '59', '_', '_', '_'],
    ['60', '61', '_', '_', '_', '62', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
    ['63', '_', '_', '_', '#', '64', '_', '_', '_', '_', '#', '65', '_', '_', '_'],
    ['66', '_', '_', '_', '#', '67', '_', '_', '_', '_', '#', '68', '_', '_', '_'],
]
across_clues = {
    1: ['AHEM', 'WHAT', 'ISAY', 'IBEG', 'AINT', 'VERB', 'PRIE', 'ESPN', 'AGOD'],
    5: ['LOTSA', 'MCJOB', 'BIGON', 'SPEED', 'MUCHO', 'GASES', 'AGITA', 'TANTO', 'ALIKE', 'NOEND'],
    10: ['MESA', 'AFRO', 'UPTO', 'EAST', 'SAGA', 'EVIL', 'ASIA', 'ENDS', 'ICED', 'APSE'],
    14: ['LORE', 'ERAS', 'OPEN', 'GAVE', 'CUED', 'ORAL', 'CLIO', 'GENE', 'EVEN', 'ANTE'],
    15: ['INONE', 'EAGLE', 'TEEUP', 'PUTTS', 'GREEN', 'HALVE', 'DIVOT', 'DRIVE', 'YARDS', 'CREEK'],
    16: ['LOOP', 'RIDE', 'OVAL', 'WAVE', 'DROP', 'WHEE', 'AUTO', 'STAR', 'SALT', 'RODE'],
    17: ['LOADEDQUESTIONS', 'INTERNALREVENUE', 'BENEFITSPACKAGE', 'SHIPSINTHENIGHT', 
         'ONEARMEDBANDITS', 'LEFTHEMISPHERES', 'KNITTINGNEEDLES', 'ACHILLESTENDONS', 
         'YOUHADMEATHELLO', 'WORDONTHESTREET'],
    20: ['RESISTANT', 'ORIENTING', 'ORIENTATE', 'DEMEANORS'],
    21: ['ANT', 'SAW', 'ADZ', 'WEE', 'TIM', 'IFI', 'UKE', 'DAB', 'TAD', 'ION'],
    22: ['UKE', 'NIT', 'TAB', 'ISA', 'PAR', 'ORT', 'FUN', 'ORE', 'TAN', 'RES'],
    23: ['OYL', 'REN', 'MOE', 'OIL', 'CEL', 'ANT', 'ASH', 'TOY', 'TAW', 'NED'],
    24: ['STEWED', 'RIPPED', 'TANKED', 'BLOTTO', 'STINKO', 'SOUSED', 'MENACE', 'LOOPED', 
         'DOWNED', 'BESOTS'],
    27: ['THEM', 'ASON', 'UNTO', 'OURS', 'AGEE', 'ASHE', 'OTIS', 'SIDE', 'RSVP', 'NATO'],
    29: ['MOTET', 'ETUDE', 'NINTH', 'ALTOS', 'ESSAY', 'NONET', 'RONDO', 'STOMP', 'IDYLL', 'GLEES'],
    32: ['EDO', 'EON', 'ERE', 'ELD', 'LAT', 'APE', 'REO', 'AVE', 'YEN', 'SEN'],
    33: ['ADO', 'APU', 'AGE', 'ITS', 'USA', 'MEN', 'WAR', 'HOG', 'IDO', 'RAN'],
    36: ['PALISADE', 'APOLOGIA', 'HEDGEROW', 'MCNAMARA', 'MANTOMAN', 'GATEPOST', 'WAVERERS', 
         'TRESPASS', 'STOCKADE', 'STIFFARM'],
    38: ['ANIMALCRACKER', 'PERPENDICULAR', 'PARALLELOGRAM', 'JURYSELECTION', 'HOLYGUACAMOLE'],
    41: ['ONELINER', 'ARBUCKLE', 'ANGELINA', 'ANACONDA', 'STREAMER', 'SENDOFFS', 'GODSPEED', 'CLEANSER'],
    42: ['ARE', 'LAY', 'ONE', 'ETC', 'BET', 'AMI', 'AND', 'ATM', 'HUH', 'OHO'],
    43: ['ERA', 'EON', 'AGE', 'EVE', 'SHE', 'FDR', 'YOU', 'RAJ', 'DOT', 'TUT'],
    44: ['TATAR', 'NOMAD', 'AARON', 'MIAMI', 'ETHEL', 'BOXER', 'ODETS', 'MIDAS', 'SWARM', 'OLDIE'],
    46: ['POLS', 'TAPS', 'MIAS'],
    50: ['MADRID', 'OVIEDO', 'ARAGON', 'TOLEDO', 'NEVADA', 'FRESNO', 'PESETA', 'PAELLA', 'ANKARA', 'OTTAWA'],
    52: ['ESP', 'SRO', 'IRS', 'EST', 'III', 'OCT', 'ERN', 'PET', 'AKA', 'IOU'],
    55: ['ROE', 'HEN', 'OVA', 'FRA', 'BIB', 'HAM', 'POT', 'DYE', 'RED', 'ALA'],
    56: ['EEL', 'OER', 'THY', 'ETC', 'TSO', 'SOS', 'EDY', 'ORD', 'IBN', 'IES'],
    57: ['DARWINWIN', 'DARWINIST', 'DARWINIAN'],
    60: ['TWOXTWENTYSEVEN', 'THREECUBEDTWICE', 'TENADDEDTOSEVEN', 'SUGARSUBSTITUTE', 
         'NINETEENLESSTWO', 'HUNDREDOVERFOUR', 'FOURTEENPLUSSIX', 'FIVETIMESELEVEN', 
         'FIFTYMINUSEIGHT', 'FIFTYLESSTWELVE'],
    63: ['ARIA', 'SANG', 'STAG', 'LEIA', 'GOIT', 'TAPS', 'LONE', 'SEUL', 'ENYA', 'ELBA'],
    64: ['FLUNG', 'SALAD', 'CABER', 'DYNES', 'THREW', 'SLUNG', 'WREST', 'QUOIT', 'ANTED', 'ADDED'],
    65: ['GLEE', 'ALTO', 'TRIO', 'IDOL', 'TINA', 'LENA', 'COMO', 'CASH', 'AMES', 'SCAT'],
    66: ['BRAS', 'LESS', 'LACY', 'ATAD', 'ALOT', 'SLIP', 'SILK', 'PINA', 'OLGA', 'LACE'],
    67: ['TEASE', 'RAGON', 'TRASH', 'TSARS', 'SPOOK', 'ROAST', 'ATONE', 'TBIRD', 'ELATE', 'AREWE'],
    68: ['HERS', 'TWIN', 'PAIR', 'EVEN', 'REDS', 'ANTI', 'RATE', 'HOLD', 'XERS', 'ROXY'],
}
down_clues = {
    2: ['HOORAY', 'HOORAH', 'IDIDIT', 'HURRAH', 'HOTDOG', 'ANACIN', 'STAIRS', 'STAPLE', 'ORELSE', 'NESTEA'],
    3: ['WEASEL', 'ERMINE', 'ALPACA', 'ORIOLE', 'AZALEA', 'OCELOT', 'MARTEN', 'MARISA', 'ISOMER', 'CORNET'],
    4: ['ICON', 'NASH', 'ERMA', 'EBAY', 'JEST', 'BLIP', 'USER', 'IDOL', 'HTTP', 'SURF'],
    5: ['LINTY', 'INANE', 'ONLOW', 'TEASE', 'TOWEL', 'AMAIN', 'WHISK', 'ERROR', 'SATED', 'LINTS'],
    6: ['RETOOK', 'ERASED'],
    7: ['ASEA', 'TEST', 'SNAG', 'AMOK', 'TALC', 'PURR', 'MEET', 'FARM', 'COMA', 'SAFE'],
    9: ['AEC', 'NRA', 'TVA', 'INS', 'SSA', 'UPI', 'KGB', 'CAA', 'ICC', 'ILO'],
    10: ['ALINE', 'SATIN', 'SHIFT', 'PUMPS', 'CHAPS', 'PREEN', 'IONIC', 'DORIC', 'SEDAN', 'RETRO'],
    11: ['HIGHTOPS', 'FOOTGEAR', 'NINEWEST', 'GALOSHES', 'TRAINERS', 'SANDALED', 'SABOTAGE', 'OPENTOED', 
         'EEEWIDTH', 'BAREFOOT'],
    12: ['RON', 'KEN', 'ANG', 'LEE', 'WES', 'TWA', 'MOE', 'CUT', 'TOD', 'TUT'],
    13: ['OPS', 'UPI', 'FCC', 'ENL', 'TED', 'PIC', 'CBC', 'LOG', 'LAG', 'OTS'],
    18: ['ONO', 'ELO', 'BAG', 'AMP', 'OWE', 'REM', 'ERA', 'PVC', 'YES', 'GIG'],
    19: ['EMPEROR', 'CAMPHOR', 'SNAPPEA', 'SHERBET', 'ORTOLAN', 'MUSTARD', 'INGESTA', 'GRANOLA', 'EGGROLL', 'ALDENTE'],
    24: ['SELMA', 'SIENA', 'YALTA', 'ALAMO', 'OZARK', 'PADUA', 'ELGIN', 'OSTIA', 'SOREL', 'ARRAS'],
    25: ['EDDY', 'DARE', 'DRAW', 'CRAG', 'MAZE', 'MEET', 'ALPS', 'LSAT', 'DEFY', 'SPOT'],
    26: ['DOE', 'ENA', 'EWE', 'EVE', 'ASE', 'HEN', 'DEN', 'SOW', 'NUN', 'DAM'],
    28: ['MASK', 'TOGA', 'SCAR', 'WART', 'VEIL', 'KILT', 'ROBE', 'TUTU', 'SARI', 'IOTA'],
    30: ['OPERA', 'ALONE', 'IDAHO', 'ARUBA', 'AUJUS', 'AMIND', 'CHESS', 'TAPAS', 'SLIDE', 'GENOA'],
    31: ['TAO', 'ROO', 'OWL', 'FIE', 'WOL', 'VIP', 'BAH', 'AGA', 'AAM'],
    34: ['DISTILL', 'ESSENCE'],
    35: ['OCTET', 'OCTAD', 'ELITE', 'BYRDS', 'MASSE', 'RANKS', 'QUEEN', 'ESSES', 'OCTAL', 'AWEEK'],
    37: ['ELAL', 'EAST', 'READ', 'SLED', 'TIME', 'DEMO', 'SHED', 'OREO', 'STEP', 'REST'],
    38: ['BORA', 'PAGO', 'NANU', 'MAHI', 'CHOP', 'CHOO', 'LIAR', 'SING', 'HEAR', 'ISAN'],
    39: ['EXECUTOR', 'DORMROOM', 'TOREADOR', 'TUITIONS', 'STREAKER', 'SCHOLARS', 'PROTESTS', 'PLEDGING', 
         'MEALPLAN', 'HARDDEAN'],
    40: ['GAT', 'ROD', 'VAL', 'REV', 'NRA', 'UZI', 'LON', 'MAE', 'LOM', 'MIA'],
    41: ['REM', 'DEW', 'ESP', 'NAP', 'BED', 'UFO', 'PSI', 'LOW', 'FOG', 'FAD'],
    45: ['RERUNS', 'ONEDGE', 'IRONIC', 'EERIER', 'HAWAII', 'SECTOR', 'SUNSET', 'GLEAMS', 'DARKEN', 'TAMALE'],
    47: ['ORIOLE', 'ALEAST', 'TOUCAN', 'PARROT', 'RAPTOR', 'AVOCET', 'PETREL', 'NESTER', 'CELTIC', 'PEAHEN'],
    48: ['LOANER', 'CABANA', 'DENTED', 'GARAGE', 'PRIMER', 'SENATE', 'POPTOP', 'BARCAR', 'TATTOO', 'USABLY'],
    49: ['SENSES', 'ISSURE', 'SENSED', 'INTUIT', 'TRUSTS', 'SEESAW', 'TASTES', 'STEERS', 'EATSIN', 'SENATE'],
    51: ['REPAY', 'STRIP', 'OWING', 'TNOTE', 'INCUR', 'RANUP', 'ARISE', 'PAYUP', 'PAYER', 'OWETO'],
    53: ['LATHE', 'PEENS', 'ROBOT', 'SWAGE', 'LADLE', 'EDGER', 'STEEL', 'RIVET', 'INGOT', 'SCRAP'],
    54: ['PIT', 'STR', 'SPA', 'STY', 'NEA', 'LEA', 'DEN', 'INN', 'SYM', 'SYD'],
    57: ['DELE', 'REDO', 'STET', 'REST', 'DESK', 'TINA', 'HIRE', 'ROSS', 'ATIP', 'THEA'],
    58: ['AQUA', 'OLAN', 'LOBE', 'PORE', 'SLOT', 'SLIT', 'IRON', 'GILL', 'ACTI', 'AERO'],
    59: ['NIGH', 'ISNT', 'NEAR', 'EZRA', 'EPOS', 'EDDA', 'ITEM', 'ROSE', 'ODES', 'IDYL'],
    60: ['SAL', 'YES', 'LEE', 'ERR', 'NRA', 'DEE', 'ACE', 'PET', 'YOU', 'RES'],
    61: ['IRA', 'BUS', 'SNO', 'CAR', 'ING', 'SNL', 'CAT', 'AOL', 'VAN', 'INT'],
    62: ['EFT', 'UKE', 'IMP', 'IMA', 'TOT', 'TOE', 'EVA', 'TAD', 'ELF', 'MAO'],
}

across2 = update_clues_with_positions(grid, across_clues, "across")
down2 = update_clues_with_positions(grid, down_clues, "down")

for clue_number, details in across2.items():
    print(f"Clue {clue_number}:")
    print(f"  Possible Words: {details[0]}")
    print(f"  Word Length: {details[1]}")
    print(f"  Start Position: {details[2]}")
    print(f"  Positions Set: {details[3]}")
    print()
for clue_number, details in down2.items():
    print(f"Clue {clue_number}:")
    print(f"  Possible Words: {details[0]}")
    print(f"  Word Length: {details[1]}")
    print(f"  Start Position: {details[2]}")
    print(f"  Positions Set: {details[3]}")
    print()
final = solve_crossword(grid2, across2, down2)
# Print the solved (or partially solved) grid
for row in final:
    # Add a space between each cell for readability
    print(" ".join(cell if cell != "." else "_" for cell in row))

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
        
        
        # cv2.imwrite("transformed.png", transformed_image)
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
        # print("Across Clues Dict:", box1_clue_dict)
        box2_clue_dict = data_clean_dict(box2_clue_dict)
        # print("Down Clues Dict:", box2_clue_dict)
        
        box3_2d_matrix = crossword_extract(box3_matrix)
        
        # # Calls to get the 2d matrix
        # # Example crossword matrix (2D array)
        # box3_2d_matrix = [
        #     ['1', '2', '3', '4', '#', '5', '6', '7', '8', '9', '#', '10', '11', '12', '13'],
        #     ['14', '_', '_', '_', '#', '15', '_', '_', '_', '_', '#', '16', '_', '_', '_'],
        #     ['17', '_', '_', '_', '18', '_', '_', '_', '_', '_', '19', '_', '_', '_', '_'],
        #     ['20', '_', '_', '_', '_', '_', '_', '_', '_', '#', '21', '_', '_', '#', '#'],
        #     ['22', '_', '_', '#', '23', '_', '_', '#', '#', '24', '_', '_', '_', '25', '26'],
        #     ['27', '_', '_', '28', '#', '#', '29', '30', '31', '_', '_', '#', '32', '_', '_'],
        #     ['#', '#', '#', '33', '34', '35', '#', '36', '_', '_', '_', '37', '_', '_', '_'],
        #     ['#', '38', '39', '_', '_', '_', '40', '_', '_', '_', '_', '_', '_', '_', '#'],
        #     ['41', '_', '_', '_', '_', '_', '_', '_', '#', '42', '_', '_', '#', '#', '#'],
        #     ['43', '_', '_', '#', '44', '_', '_', '_', '45', '#', '#', '46', '47', '48', '49'],
        #     ['50', '_', '_', '51', '_', '_', '#', '#', '52', '53', '54', '#', '55', '_', '_'],
        #     ['#', '#', '56', '_', '_', '#', '57', '58', '_', '_', '_', '59', '_', '_', '_'],
        #     ['60', '61', '_', '_', '_', '62', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
        #     ['63', '_', '_', '_', '#', '64', '_', '_', '_', '_', '#', '65', '_', '_', '_'],
        #     ['66', '_', '_', '_', '#', '67', '_', '_', '_', '_', '#', '68', '_', '_', '_']
        # ]
        
        # # Print the box3 matrix for debugging
        # for row in box3_2d_matrix:
        #     print(row)
            
        answer_matrix = generate_answer_matrix(box3_2d_matrix)
        
        
        # # Testing for obtaining pattern p for 57 across for debugging
        # clue_number = 57
        # pattern = find_pattern_across(box3_2d_matrix, answer_matrix, clue_number)
        # print(f"Pattern for clue {clue_number} across: {pattern}")

        # # Testing for obtaining pattern p for 57 down for debugging        
        # clue_number = 57
        # pattern = find_pattern_down(box3_2d_matrix, answer_matrix, clue_number)
        # print(f"Pattern for clue {clue_number} across: {pattern}")        
        
        # # Testing for getting a response answer array for a clue          
        # clue = "Beg-pardon-..."
        # pattern = "????"
        # length = 4

        # answers = fetch_crossword_answers(clue, pattern, length)
        # print("Possible Answers:", answers)        

        
        # Testing to get the Across possible answers
        across_answers = generate_possible_clue_answers(
            clue_dict=box1_clue_dict,
            crossword=box3_2d_matrix,
            answer_matrix=answer_matrix,
            direction="across"
        )

        # Testing to get the Down possible answers
        down_answers = generate_possible_clue_answers(
            clue_dict=box2_clue_dict,
            crossword=box3_2d_matrix,
            answer_matrix=answer_matrix,
            direction="down"
        )
         # Print the crossword puzzle grid
        print("\nCrossword Puzzle Grid:")
        for row in box3_2d_matrix:
            print(" ".join(row))
        # Print the answer matrix for debugging
        print('answer matrix')
        for row in answer_matrix:
            print(row)
        # Print all possible answers for across clues
        print("\nPossible Answers for Across Clues:")
        for clue_number, answers in across_answers.items():
            print(f"Clue {clue_number}: {answers}")

        # Print all possible answers for down clues
        print("\nPossible Answers for Down Clues:")
        for clue_number, answers in down_answers.items():
            print(f"Clue {clue_number}: {answers}")

        # # Outputting the Across and Down possible answer results to terminal
        # print("Possible Across Answers:", across_answers)
        # print("Possible Down Answers:", down_answers)
        
        # Converting to hex for transport to output but honestly questioning whether bytes would be better
        box1_across_hex = image_to_hex(box1_across)
        box2_down_hex = image_to_hex(box2_down)
        box3_matrix_hex = image_to_hex(box3_matrix)
        # box3_2d_matrix = image_to_hex(box3_2d_matrix)
       
        return jsonify({
            "original_image": original_image_hex,
            "transformed_image": transformed_image_hex,
            "corners": corners.tolist(),  # Return corner values for further processing if needed
            "box1_across": box1_across_hex,
            "box2_down": box2_down_hex,
            "box3_matrix": box3_matrix_hex,
            # "box3_2d_matrix": box3_2d_matrix
        })
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
