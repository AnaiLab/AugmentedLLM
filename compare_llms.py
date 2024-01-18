from modules.models import OpenAIChatModel, HuggingfaceRemoteModel
from config import Config
from questions import Questions
from datetime import datetime

# Edit to change where responses are saved
output_dir = './llm_responses/'


models = []

# Open AI models
openaiSystemPrompt = 'You are a helpful assistant. Answer each multiple choice question to the best of your knowledge. Only reply with the letter for the correct answer. Your answer should be 1 character.'

gpt35turbo = OpenAIChatModel(Config.OPENAI_KEY, 'gpt-3.5-turbo', openaiSystemPrompt)
models.append(gpt35turbo)

gpt4 = OpenAIChatModel(Config.OPENAI_KEY, 'gpt-4', openaiSystemPrompt)
models.append(gpt4)

# HF Models
mistral = HuggingfaceRemoteModel(Config.HUGGINGFACE_KEY, 'mistralai/Mistral-7B-v0.1')
models.append(mistral)

for model in models:
    timestamp = datetime.now().strftime('%d-%m-%y-%H:%M')
    filename = output_dir + model.model + '-' + timestamp
    responses = ''

    # Loop over the questions and generate a response for each
    for question in Questions:
        response = model.query({'inputs': question.question + '\nCorrect answer is'})

        responses += 'Question: ' + question.question + '\nResponse: ' + response + '\nCorrect answer: ' + question.answer + '\n-----------------------------------\n\n\n' 

        
    # Write output to file
    with open("test.txt", "w") as f:
        f.write(responses)
