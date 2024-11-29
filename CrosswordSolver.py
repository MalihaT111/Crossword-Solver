import random
import string

def calculate_local_conflicts(grid, word, positions):
    """
    Calculate the number of conflicts caused by placing a word at specified positions.

    Args:
        grid (list[list[str]]): The current crossword grid.
        word (str): The word being placed.
        positions (list[tuple]): List of (row, col) positions where the word will be placed.

    Returns:
        int: The number of conflicts caused by placing the word.
    """
    conflicts = 0
    for (row, col), letter in zip(positions, word):
        if grid[row][col] != '_' and grid[row][col] != letter:
            conflicts += 1  # Count conflict if the cell is already filled with a different letter
    return conflicts


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


# now i have a function to go through the dictionary and append the proper things 
def update_clues_with_positions(grid, clues, direction, max_conflicts=10):
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
        
        # Skip clues with word lengths greater than 14
        if word_length > 14:
            continue
        
        # Count conflicts for the current grid state
        conflicts = sum(
            1 for row, col in positions_set if grid[row][col] != '_' and grid[row][col] not in [word[i] for word in words for i in range(len(word))]
        )
        
        # Exclude clues with too many conflicts
        if conflicts > max_conflicts:
            continue
        
        # Update the clue value with the new information
        updated_clues[clue_number] = (words, word_length, start_pos, positions_set)
    return updated_clues

def place_word(grid, word, positions):
    """
    Places a word in the grid at the specified positions.

    Args:
        grid (list[list[str]]): The crossword grid.
        word (str): The word to place.
        positions (list[tuple]): List of (row, col) positions.
    """
    for (row, col), letter in zip(positions, word):
        grid[row][col] = letter


def remove_word(grid, positions):
    """
    Removes a word from the grid by resetting its positions to '_'.

    Args:
        grid (list[list[str]]): The crossword grid.
        positions (list[tuple]): List of (row, col) positions.
    """
    for row, col in positions:
        grid[row][col] = '_'

def count_conflicts(grid, word, positions):
    """
    Count the number of conflicts a word causes when placed in the grid.

    Args:
        grid (list[list[str]]): The crossword grid.
        word (str): The word to place.
        positions (list[tuple]): List of (row, col) positions for the word.

    Returns:
        int: The number of conflicts.
    """
    conflicts = 0
    for (row, col), letter in zip(positions, word):
        if grid[row][col] != '_' and grid[row][col] != letter:
            conflicts += 1
    return conflicts

    
def select_least_conflicted_word(words, grid, positions):
    """
    Select the word with the fewest conflicts and the best impact on future variables.

    Args:
        words (list[str]): List of possible words for a clue.
        grid (list[list[str]]): The crossword grid.
        positions (list[tuple]): List of (row, col) positions for the word.

    Returns:
        str: The best word to place. Returns None if no word fits.
    """
    best_word = None
    min_score = float('inf')

    for word in words:
        conflicts = count_conflicts(grid, word, positions)
        score = conflicts  # Start with conflicts as the base score

        # Adjust score based on additional heuristics (e.g., domain reduction)
        for (row, col), letter in zip(positions, word):
            if grid[row][col] == '_':  # Only consider empty cells
                # Reward filling intersections that constrain future options
                score -= 1

        if score < min_score:
            min_score = score
            best_word = word

    return best_word



def sort_clues_by_word_length(clues):
    """
    Sort clues by word length in ascending order.

    Args:
        clues (dict): Dictionary of clues.

    Returns:
        list: Sorted list of clues (clue_number, details) by word length.
    """
    return sorted(clues.items(), key=lambda item: item[1][1]) 

    
def solve_crossword_as_csp(grid, across_clues, down_clues):
    """
    Solve the crossword puzzle using the least-conflicted word strategy.

    Args:
        grid (list[list[str]]): The crossword grid.
        across_clues (dict): Across clues with positions and domains.
        down_clues (dict): Down clues with positions and domains.

    Returns:
        list[list[str]]: The solved crossword grid or None if unsolvable.
    """
    def backtrack(variables):
        # If all variables are assigned, the puzzle is solved
        if not variables:
            return True

        # Choose the next variable (shortest word first)
        clue_number, details = variables.pop(0)
        words, word_length, start_pos, positions = details

        # Select the least conflicted word
        word = select_least_conflicted_word(words, grid, positions)
        
        if word is None:
            # Backtrack if no word fits
            variables.insert(0, (clue_number, details))
            return False

        # Place the word in the grid
        place_word(grid, word, positions)

        # Recursively solve the remaining puzzle
        if backtrack(variables):
            return True
            
        # Remove the word if it leads to conflicts
        remove_word(grid, positions)
        variables.insert(0, (clue_number, details))
        return False

    # Combine across and down clues, and sort by word length
    variables = list(sort_clues_by_word_length({**across_clues, **down_clues}))

    # Solve using backtracking
    if backtrack(variables):
        return grid
    else:
        return None

def reevaluate_and_correct(grid, clues, direction):
    """
    Reevaluates and corrects invalid words in the solved crossword grid.
    Leaves invalid words (not in the word list) as empty cells.

    Args:
        grid (list[list[str]]): The solved crossword grid.
        clues (dict): Dictionary of clues with positions and word lists.
        direction (str): "across" or "down".

    Returns:
        list[list[str]]: The corrected crossword grid.
    """
    def calculate_mismatch(word1, word2):
        """Calculate the number of mismatched letters between two words."""
        return sum(1 for a, b in zip(word1, word2) if a != b)

    def allowed_mismatches(word_length):
        """Determine the maximum number of mismatches allowed based on word length."""
        if word_length <= 4:
            return 1
        elif word_length <= 10:
            return 2
        else:
            return 3

    for clue_number, details in clues.items():
        words, word_length, start_pos, positions = details

        # Extract the current word from the grid
        current_word = ''.join(grid[row][col] for row, col in positions)

        # Skip if the current word is valid
        if current_word in words:
            continue

        # Determine the allowed mismatches for this word length
        max_mismatches = allowed_mismatches(len(current_word))

        # Find the closest matching word
        best_match = None
        min_mismatches = float('inf')

        for word in words:
            if len(word) == len(current_word):  # Ensure the word lengths match
                mismatches = calculate_mismatch(current_word, word)
                if mismatches < min_mismatches:
                    min_mismatches = mismatches
                    best_match = word

        # Replace the word if mismatches are within the allowed range
        if best_match and min_mismatches <= max_mismatches:
            for (row, col), letter in zip(positions, best_match):
                grid[row][col] = letter
        else:
            # Leave the cells empty for invalid words
            for row, col in positions:
                grid[row][col] = '_'

    return grid

def count_differences(solved_grid, real_answer):
    """
    Count the number of differences between the solved grid and the real answer.

    Args:
        solved_grid (list[list[str]]): The solved crossword grid.
        real_answer (list[list[str]]): The real crossword answer grid.

    Returns:
        int: The count of differing cells.
    """
    differences = 0
    for row_idx in range(len(real_answer)):
        for col_idx in range(len(real_answer[row_idx])):
            if solved_grid[row_idx][col_idx] != real_answer[row_idx][col_idx]:
                differences += 1
    return differences

