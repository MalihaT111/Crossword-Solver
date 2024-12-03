from flask import Flask, request, jsonify, render_template
import io
import requests
import urllib
from bs4 import BeautifulSoup
from CrosswordSolver import *
from ImageProcessing import * 
from FeatureExtraction import *

import subprocess
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import time
from multiprocessing import Pool, cpu_count

app = Flask(__name__)

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
    
def generate_possible_clue_answers(clue_dict, crossword, answer_matrix, direction, specific_pass=False):
    """
    Generate a dictionary of possible answers for each clue.

    Parameters:
        clue_dict (dict): Dictionary of clues (clue_number: clue_text).
        crossword (list[list[str]]): The crossword grid.
        answer_matrix (list[list[str]]): The partially filled crossword grid with known letters.
        direction (str): Direction of clues ("across" or "down").
        specific_pass (bool): If True, use a more specific query strategy with known letters.

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

            # Decide which function to use based on the pass type
            if specific_pass:
                # Use the updated function for a refined query
                possible_answers = fetch_specific(clue_text, pattern, len(pattern))
            else:
                # Use the original function for a general query
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

def solve(number_grid, empty_grid, across_clues, down_clues): 
    new_across = update_clues_with_positions(number_grid, across_clues, "across")
    new_down = update_clues_with_positions(number_grid, down_clues, "down")
    solved_grid = solve_crossword_as_csp(empty_grid, new_across, new_down)
    if solved_grid:
        solved_grid = reevaluate_and_correct(solved_grid, new_across, "across")
        solved_grid = reevaluate_and_correct(solved_grid, new_down, "down")
    return solved_grid

# Function to get a dictionary of clue numbers to an array of clue possible answers
def generate_request_calls(clue_dict, crossword, answer_matrix, direction):
    """
    Generate a list of request call parameters for crossword clue processing.

    Parameters:
        clue_dict (dict): Dictionary of clues (clue_number: clue_text).
        crossword (list[list[str]]): The crossword grid.
        answer_matrix (list[list[str]]): The answer matrix.
        direction (str): Direction of clues ("across" or "down").

    Returns:
        list: A list of tuples, where each tuple contains (clue_number, clue_text, pattern, length).
    """
    request_calls = []

    for clue_number, clue_text in clue_dict.items():
        try:
            # Determine the pattern based on the direction
            if direction == "across":
                pattern = find_pattern_across(crossword, answer_matrix, clue_number)
            elif direction == "down":
                pattern = find_pattern_down(crossword, answer_matrix, clue_number)
            else:
                raise ValueError("Invalid direction. Use 'across' or 'down'.")

            # Append the parameters for the request call
            request_calls.append((clue_number, clue_text, pattern, len(pattern)))
        except Exception as e:
            print(f"Error generating request call for clue {clue_number}: {e}")

    return request_calls

def call_fetch_worker(clue_text, pattern, length):
    """
    Call the fetch_worker.py script to fetch possible answers for a clue.
    """
    try:
        result = subprocess.run(
            ["python3", "fetch_worker.py", clue_text, pattern, str(length)],
            text=True, capture_output=True, check=True
        )
        # Parse the JSON output
        return json.loads(result.stdout)["answers"]
    except subprocess.CalledProcessError as e:
        print(f"Error in fetch_worker.py: {e.stderr}")
        return []

def fetch_clue_answers_multithreaded(request_calls):
    """
    Fetch crossword answers for clues using multithreading with subprocess calls.
    """
    possible_clue_answers = {}

    # Multithreaded execution of subprocess calls
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(call_fetch_worker, clue_text, pattern, length): clue_number
            for clue_number, clue_text, pattern, length in request_calls
        }

        for future in concurrent.futures.as_completed(futures):
            clue_number = futures[future]
            try:
                possible_clue_answers[clue_number] = future.result()
            except Exception as e:
                print(f"Error processing clue {clue_number}: {e}")

    return possible_clue_answers

# Function to test the accuracy and efficiency of crossword solver code
def test_solver(solver_function, puzzle):
    """
    Tests the accuracy and efficiency of a crossword solver.

    Parameters:
    - solver_function (function): The crossword-solving function.
    - puzzle (list of list of str): The crossword grid as a 2D list. Empty cells are "" or None.
    - solution (list of list of str): The correct solution grid as a 2D list.

    Returns:
    - dict: A dictionary containing accuracy, time taken, and solver output.
    
    Accuracy: Use the percentage of correctly filled cells (or letters) in the crossword 
    grid compared to the expected solution.

    Formula:
    Accuracy = (Number_of_Correctly_Filled_Cells/Total_Number_of_Cells)x100

    Efficiency: Measure the total time taken to solve the puzzle and analyze bottlenecks 
    using profiling tools (like cProfile or timeit). Bottlenecks could include:

    - Parsing clues
    - Grid placement logic
    - Dictionary lookups or constraint satisfaction algorithms
        
    """

    solution = [
            ['A', 'H', 'E', 'M', '#', 'L', 'O', 'T', 'S', 'A', '#', 'A', 'F', 'R', 'O'],
            ['L', 'O', 'R', 'E', '#', 'I', 'N', 'O', 'N', 'E', '#', 'L', 'O', 'O', 'P'],
            ['C', 'O', 'M', 'M', 'O', 'N', 'F', 'R', 'A', 'C', 'T', 'I', 'O', 'N', 'S'],
            ['O', 'R', 'I', 'E', 'N', 'T', 'I', 'N', 'G', '#', 'A', 'N', 'T', '#', '#'],
            ['T', 'A', 'N', '#', 'O', 'Y', 'L', '#', '#', 'S', 'T', 'E', 'W', 'E', 'D'],
            ['T', 'H', 'E', 'M', '#', '#', 'M', 'O', 'T', 'E', 'T', '#', 'E', 'D', 'O'],
            ['#', '#', '#', 'A', 'D', 'O', '#', 'P', 'A', 'L', 'I', 'S', 'A', 'D', 'E'],
            ['#', 'B', 'A', 'S', 'I', 'C', 'G', 'E', 'O', 'M', 'E', 'T', 'R', 'Y', '#'],
            ['R', 'O', 'C', 'K', 'S', 'T', 'A', 'R', '#', 'A', 'R', 'E', '#', '#', '#'],
            ['E', 'R', 'A', '#', 'T', 'A', 'T', 'A', 'R', '#', '#', 'P', 'O', 'L', 'S'],
            ['M', 'A', 'D', 'R', 'I', 'D', '#', '#', 'E', 'S', 'P', '#', 'R', 'O', 'E'],
            ['#', '#', 'E', 'E', 'L', '#', 'D', 'A', 'R', 'W', 'I', 'N', 'I', 'A', 'N'],
            ['S', 'I', 'M', 'P', 'L', 'E', 'E', 'Q', 'U', 'A', 'T', 'I', 'O', 'N', 'S'],
            ['A', 'R', 'I', 'A', '#', 'F', 'L', 'U', 'N', 'G', '#', 'G', 'L', 'E', 'E'],
            ['L', 'A', 'C', 'Y', '#', 'T', 'E', 'A', 'S', 'E', '#', 'H', 'E', 'R', 'S']
        ]

    # Measure efficiency
    start_time = time.time()
    solver_output = solver_function(puzzle)
    end_time = time.time()
    time_taken = end_time - start_time

    # Measure accuracy
    total_cells = 15^2
    
    # Measure number of differences
    differences = count_differences(solver_output, solution)

    accuracy = ((total_cells-differences) / total_cells) * 100 if total_cells > 0 else 0


    # Return results
    return {
        "accuracy": accuracy,
        "differences": differences,
        "time_taken": time_taken,
        "solver_output": solver_output
    }


@app.route('/', methods=['GET','POST'])
def upload_image():
    if request.method == "GET":
        # Render the upload form
        return render_template("index.html")
    if request.method == "POST":
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
            across_box, down_box, crossword = extract_boxes(transformed_image)
            
            # Extract text from box1_across and box2_down for the across and down clues
            across_text = extract_text(across_box)
            down_text = extract_text(down_box)
                
            initial_across_hints = extract_clues(across_text)
            initial_down_hints = extract_clues(down_text)
            
            across_hints = data_clean_dict(initial_across_hints)
            down_hints = data_clean_dict(initial_down_hints)
            
            number_grid = crossword_extract(crossword) 

                
            empty_grid = generate_answer_matrix(number_grid)
            
            # # Testing to get the Across possible answers
            # across_clues = generate_possible_clue_answers(
            #     clue_dict=across_hints,
            #     crossword=number_grid,
            #     answer_matrix=empty_grid,
            #     direction="across"
            # )

            # # Testing to get the Down possible answers
            # down_clues = generate_possible_clue_answers(
            #     clue_dict=down_hints,
            #     crossword=number_grid,
            #     answer_matrix=empty_grid,
            #     direction="down"
            # )
            
            request_calls1 = generate_request_calls(across_hints, number_grid, empty_grid, "across")
            # print(request_calls1) 
            across_clues = fetch_clue_answers_multithreaded(request_calls1)

            request_calls2 = generate_request_calls(down_hints, number_grid, empty_grid, "down")
            # print(request_calls2) 
            down_clues = fetch_clue_answers_multithreaded(request_calls2)
            
            
            # # Converting to hex for transport to output but honestly questioning whether bytes would be better
            # box1_across_hex = image_to_hex(across_box)
            # box2_down_hex = image_to_hex(down_box)
            # box3_matrix_hex = image_to_hex(crossword)

            solved_grid = solve(number_grid, empty_grid, across_clues, down_clues)

            test_results = test_solver(solve, number_grid)
            
            

            solved_grid_html = "<br>".join([" ".join(row) for row in solved_grid])

            return f"<h1>Solved Crossword</h1><pre>{solved_grid_html}</pre>"
        except Exception as e:
            return f"Error processing the image: {str(e)}", 500
        
        # if solved_grid:
        #     # print("\nSolved Crossword Grid:")
        #     # for row in solved_grid:
        #     #     print(" ".join(row))

        #     return jsonify({
        #         "original_image": original_image_hex,
        #         "transformed_image": transformed_image_hex,
        #         "corners": corners.tolist(),  # Return corner values for further processing if needed
        #         "box1_across": box1_across_hex,
        #         "box2_down": box2_down_hex,
        #         "box3_matrix": box3_matrix_hex,
        #     })
        
        # except ValueError as e:
        #     return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
