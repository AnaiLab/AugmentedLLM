import re

def grade(answer_attempt: str,true_answer: str):
    """Grades a MCQ. Returns 1 if correct, 0 if incorrect, -1 if the answer does not follow the expected pattern"""
    # Check the strings are real
    if(len(answer_attempt) < 1 or len(true_answer) <1):
        return -1

    # Check the format
    regex = f'^[A-Za-z]\\.?$'
    if(not re.fullmatch(regex, answer_attempt) or not re.fullmatch(regex, true_answer)):
        return -1

    # See if they match
    if answer_attempt[0].lower() == true_answer[0].lower():
        return 1
    
    return 0


def write_to_file(file_path, s):
    """
    Writes the given text to a file. Appends if the file exists, otherwise creates a new file.
    """
    with open(file_path, 'a') as file:
        file.write(s + "\n")


