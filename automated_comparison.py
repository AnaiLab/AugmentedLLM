from config import Config
import os
import lmql
from questions import Questions
from utils import utils
from datetime import datetime

# Set openai key as expected by lmql
os.environ['OPENAI_API_KEY'] = Config.OPENAI_KEY

# Edit to change where responses are saved
output_dir = './llm_responses/'

models = []

# uncomment out whichever models you would not like to include, or specify your own
# GPT 3.5
# models.append(lmql.model("openai/gpt-3.5-turbo"))

# GPT-4
# models.append(lmql.model("openai/gpt-4"))

if(len(models) == 0):
    print("Please uncomment at least one model for testing!")
    exit(1)

for model in models:
    timestamp = datetime.now().strftime('%d-%m-%y-%H:%M')
    filename = output_dir + model.model_identifier.replace('/','-') + '-' + timestamp
    
    responses = ''
    correct = 0
    total = len(Questions)

    for question in Questions:
        answer = model.generate_sync(question.question, max_tokens=1).strip()
        grade = utils.grade(answer, question.answer)

        if grade == -1:
            print(f'There was an error parsing the following\n Given answer: {answer}\nCorrect answer: {question.answer}\nIt will be excluded from the total and must be graded manually to be included.') 
            total -= 1
            continue

        # Grade is 0 if wrong, 1 if right
        correct += grade

        responses += f'Question: {question.question}\n Given answer: {answer}\n Correct answer: {question.answer}\n\n\n'

    if(total == 0):
        score = 0
    else:
        score = correct / total

    # Write output to file as well as terminal
    print(f'{model.model_identifier}: {score * 100}')
    utils.write_to_file(filename, responses)