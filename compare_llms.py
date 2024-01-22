from modules.models import OpenAIChatModel, HuggingfaceRemoteModel
from config import Config
from questions import Questions
from datetime import datetime

# Edit to change where responses are saved
output_dir = './llm_responses/'


models = []

# TODO: add augmented model

# Open AI models
openaiSystemPrompt = 'You are a helpful assistant. Answer each multiple choice question to the best of your knowledge. Only reply with the letter for the correct answer. Your answer should be 1 character.'

# GPT 3.5 TURBO
# models.append(OpenAIChatModel(Config.OPENAI_KEY, 'gpt-3.5-turbo', openaiSystemPrompt))

# GPT 4
# models.append(OpenAIChatModel(Config.OPENAI_KEY, 'gpt-4', openaiSystemPrompt))

# HF Models
# TODO: make these specific huggingfaceremote models

# Mistral 7b
# models.append(HuggingfaceRemoteModel(Config.HUGGINGFACE_KEY, 'mistralai/Mistral-7B-v0.1'))

# Openchat 3.5
# models.append(HuggingfaceRemoteModel(Config.HUGGINGFACE_KEY, 'openchat/openchat-3.5-0106'))

if(len(models) == 0):
    print("Please uncomment at least one model for testing")
    exit(1)

for model in models:
    timestamp = datetime.now().strftime('%d-%m-%y-%H:%M')
    filename = output_dir + model.model + '-' + timestamp
    responses = ''

    # Loop over the questions and generate a response for each
    for question in Questions:
        
        response = model.query({'inputs': question.question + '\nCorrect answer is:'})
        responses += 'Question: ' + question.question + '\nResponse: ' + response + '\nCorrect answer: ' + question.answer + '\n-----------------------------------\n\n\n' 

        
    # Write output to file
    with open(filename, "w") as f:
        f.write(responses)
