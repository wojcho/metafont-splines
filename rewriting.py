import json

from knots_to_svg import from_knots_to_svg_one_curve

ALPHABET_TO_KNOTS: dict[str, list[list[tuple[float, float]]]] = {
    # "|": [[(0, 0.2), (0.2, 0.6), (0, 1)]],
    # "_": [[(0, 0), (1.2, 0)]],

    "p": [[(0.5, 1), (0.5, 0.6), (1, 0.2)], [(1, 0.2), (0.8, 0.6), (1, 1)]],
    "b": [[(0.5, 0.2), (0.5, 0.6), (0.3, 1)], [(0.3, 1), (1, 1), (1, 0.2)]],
    "t": [[(0.3, 1), (0.7, 0.3), (0.9, 0.2), (1, 0.4), (1, 0.8), (0.6, 1)]],
    "d": [[(0.3, 1), (0.7, 1.2), (1, 1.5)], [(1, 1.5), (0.8, 0.6), (0.8, 0.2)]],
    "k": [[(0.5, 0.2), (0.6, 0.7), (0.5, 1.2), (0.3, 1.5)], [(0.3, 1.5), (0.6, 1.5), (0.9, 1.4), (1, 1), (1, 0.2)]],
    "g": [[(0.2, 1), (0.5, 1), (0.9, 0.5), (1, 0.2)], [(1, 0.2), (1, 1.5)]],
    "m": [[(0.8, 0.2), (1, 0.6), (0.9, 0.9), (0.8, 1), (0.4, 1), (0.2, 1)]],
    "n": [[(1, 0.2), (1, 1), (1, 1.3), (0.7, 1.4), (0.2, 1.3)]],
    "l": [[(0.4, 0.8), (0.6, 0.4), (1, 0.2)], [(1, 0.2), (0.5, 1), (0.5, 1.4), (1, 1)]],
    "r": [[(1, 0.2), (0.4, 1), (0.4, 1.5)], [(0.4, 1.5), (0.7, 1), (1, 1)]],
    "j": [[(1, 0.2), (0.5, 0.5), (0.3, 1)], [(0.3, 1), (0.7, 0.8), (0.9, 1.4), (0.5, 1.5), (0.2, 1.5)]],
    "h": [[(0.4, 0.8), (0.4, 1), (0.8, 1), (1, 1.2), (1, 1.5)]],
    "s": [[(0.8, 0.2), (0.8, 0.6), (0.3, 1)], [(0.3, 1), (0.6, 1), (0.8, 1), (1, 1.2), (1, 1.5)]],
    "v": [[(1, 1), (0.8, 0.6), (1, 0.2)], [(1, 1), (0.8, 1), (0.5, 1), (0.2, 1.5)]],

    "a": [[(0, -0.5), (0.2, -0.3), (0.4, -0.5)], [(0.4, -0.5), (0.6, -0.3), (1, -0.5)]],
    "e": [[(0, -0.3), (0.5, -0.3), (1, -0.5)], [(0.4, -0.6), (0.5, -0.4)]],
    "i": [[(0, -0.3), (0.5, -0.3), (1, -0.5)]],
    "o": [[(0, -0.5), (0.2, -0.5), (0.4, -0.3)], [(0.4, -0.3), (0.6, -0.5), (1, -0.5)]],
    "u": [[(0, -0.3), (0.5, -0.5), (1, -0.5)]],
}

CONSONANTS = "pbtkdmnlrjhsv"
VOWELS = "aeiou"
ALLOWED_CHARACTERS = f"{CONSONANTS}{VOWELS} "

HORIZONTAL_CHARACTER_LENGTH = 1.2
VERTICAL_VOWEL_HEIGHT = 0.2

def translate_polylines(polylines: list[list[tuple[float, float]]], translation: tuple[float, float]) -> list[list[tuple[float, float]]]:
    dx, dy = translation
    return [
        [(x + dx, y + dy) for x, y in polyline]
        for polyline in polylines
    ]

def text_to_knots_lists(text: str) -> list[list[tuple[float, float]]]:
    # Check whether text contains only characters from allowed set, otherwise raise
    if not (set(text) <= set(ALLOWED_CHARACTERS)):
        not_whitelisted = set(text) - set(ALLOWED_CHARACTERS)
        raise ValueError(f"Text contains disallowed characters: {not_whitelisted}")

    knots_lists: list[list[tuple[float, float]]] = []

    # Split by spaces, count consonants to obtain lengths of words, for drawing of upper horizontal line and vertical lines
    word_consonant_amounts: list[int] = [
        sum(1 for ch in word if ch in CONSONANTS)
        for word in text.split()
    ]

    # Draw upper line for all words, assuming any amount of spaces between words is treated as one
    current_position: tuple[float, float] = (0, 0)
    for word_length in word_consonant_amounts:
        word_knotspace_visual_length = word_length * HORIZONTAL_CHARACTER_LENGTH
        knots_lists.append([current_position, (current_position[0] + word_knotspace_visual_length, current_position[1])])
        current_position = (current_position[0] + (word_length + 1) * HORIZONTAL_CHARACTER_LENGTH, current_position[1])
    
    # Draw vertical lines for all word parts
    current_position = (0, 0)
    for word_length in word_consonant_amounts:
        for i in range(word_length):
            current_vertical_line = translate_polylines([[(0, 0.2), (0.2, 0.6), (0, 1)]], current_position)
            knots_lists.extend(current_vertical_line)
            current_position = (current_position[0] + HORIZONTAL_CHARACTER_LENGTH, current_position[1])
        current_position = (current_position[0] + HORIZONTAL_CHARACTER_LENGTH, current_position[1])

    # Draw consonants and vowels
    current_position = (0, 0)
    for index, char in enumerate(text):
        if char == " " and (index == 0 or text[index - 1] != " "):
            current_position = (current_position[0] + HORIZONTAL_CHARACTER_LENGTH, current_position[1])
        if char in VOWELS:
            current_vowel = translate_polylines(ALPHABET_TO_KNOTS[char], current_position)
            knots_lists.extend(current_vowel)
            current_position = (current_position[0], current_position[1] - VERTICAL_VOWEL_HEIGHT)
            if (index + 1) < len(text) and text[index + 1] not in VOWELS:
                current_position = (current_position[0] + HORIZONTAL_CHARACTER_LENGTH, current_position[1])
        if char in CONSONANTS:
            current_position = (current_position[0], 0)
            current_consonant = translate_polylines(ALPHABET_TO_KNOTS[char], current_position)
            knots_lists.extend(current_consonant)
            if (index + 1) < len(text) and text[index + 1] not in VOWELS:
                current_position = (current_position[0] + HORIZONTAL_CHARACTER_LENGTH, current_position[1])

    return knots_lists

def text_to_svg_path_d(text: str) -> list[MetafontOutlineCenterline]:
    knots_lists: list[list[tuple[float, float]]] = text_to_knots_lists(text)
    outline_centerlines_lists: list[MetafontOutlineCenterline] = [from_knots_to_svg_one_curve(knots) for knots in knots_lists]
    return outline_centerlines_lists

if __name__ == "__main__":
    outline_centerlines_lists = text_to_svg_path_d("sauple tekt")
    pseudo_json = []
    for outline_centerlines in outline_centerlines_lists:
        pseudo_json.append({ "centerline": outline_centerlines.curve_path, "outlines": outline_centerlines.outline_paths })
    print(json.dumps(pseudo_json, indent=4))
