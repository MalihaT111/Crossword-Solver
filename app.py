from flask import Flask, request, jsonify
import cv2
import numpy as np
import io
from PIL import Image, ImageEnhance, ImageFilter
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

# Function to use ocr to extract string of clue information from a box
def extract_text(image):
    # Open the image file
    pil_image = Image.fromarray(image)
    
    # Use Tesseract to extract text
    return pytesseract.image_to_string(pil_image)

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

def solve_crossword(crossword, answer_matrix, across_clues, down_clues):
    """
    Solves the crossword puzzle using backtracking.

    Parameters:
        crossword (list[list[str]]): The crossword grid with clue numbers and '#' for black spaces.
        answer_matrix (list[list[str]]): The answer grid to fill.
        across_clues (dict): Dictionary of across clues with possible answers.
        down_clues (dict): Dictionary of down clues with possible answers.

    Returns:
        bool: True if the crossword is solved, False otherwise.
    """
    # Find the next clue to solve (heuristic: fewest options first)
    clue_number, direction, possible_answers = get_next_clue(across_clues, down_clues)

    # Base case: No clues left, puzzle solved
    if not clue_number:
        return True

    # Get the current pattern and verify possible answers
    if direction == "across":
        pattern_func = find_pattern_across
    else:
        pattern_func = find_pattern_down

    pattern = pattern_func(crossword, answer_matrix, clue_number)

    for answer in possible_answers:
        if is_valid_answer(answer_matrix, clue_number, answer, direction, crossword):
            # Place the answer in the grid
            place_answer(answer_matrix, clue_number, answer, direction)

            # Recur to solve the rest of the puzzle
            if solve_crossword(crossword, answer_matrix, across_clues, down_clues):
                return True

            # Undo the placement (backtracking)
            remove_answer(answer_matrix, clue_number, answer, direction)

    return False

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
        
        # # Print the answer matrix for debugging
        # for row in answer_matrix:
        #     print(row)
        
        
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
