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
solved_grid = solve_crossword_as_csp(grid2, across2, down2)
if solved_grid:
    solved_grid = reevaluate_and_correct(solved_grid, across2, "across")
    solved_grid = reevaluate_and_correct(solved_grid, down2, "down")

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

# Real answer grid (example provided in the question)
real_answer = [
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
print(count_differences(real_answer, solved_grid))

if solved_grid:
    print("\nSolved Crossword Grid:")
    for row in solved_grid:
        print(" ".join(row))
print("\nAnswer Key")
for row in real_answer:
    print(" ".join(row))
