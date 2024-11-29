
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