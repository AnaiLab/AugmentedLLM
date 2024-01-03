from openai import OpenAI
import requests
import string
import re

# TODO: Have augmented model inherit from this as well

"""Generic model class. Not to be instantiated, use a child class instead"""
class Model(object):
    abstractError = "Please do not instantiate the abstract class, use a child class."

    """Answers a multiple choice question passed in as the query parameter. Pass in the number of answer choices (presumed A,B,...)"""
    def query(self, query: str, numAnswers: int) -> str:
        raise NotImplementedError(self.abstractError)
    
    """Generates an answer to the MCQ, and returns 1 if the answer is correct, 0 if it is incorrect, and -1 if it does not follow the expected format"""
    def gradeMCQ(self, query: str, numAnswers: int, answer: str) -> int:
        result = self.query(query)

        regex = self._getMCQRegex(numAnswers)

        if not (self._checkString(result,regex)):
            return -1
        elif result == answer:
            return 1
        else:
            return 0
    
    """Gets the regex for an MCQ with the desired number of answer options"""
    def __getMCQRegex(n: int):
        # If out of range just allow all the options
        if n < 1 or n > 26:
            n = 26

        upper_limit = string.ascii_uppercase[n - 1]
        lower_limit = string.ascii_lowercase[n - 1]

        # Construct the regex
        regex = f'^[A-{upper_limit}a-{lower_limit}]\\.?$'

        return regex
    
    """Check a string against the supplied regex pattern"""
    def __checkString(s: str, regex: str) -> bool:
        r = re.compile(regex)
        return bool(r.fullmatch(s))




"""Client for interacting with HuggingFace Models via inference API"""
class HuggingfaceRemoteModel(Model):
    def __init__(self, apiKey: str, model: str):
        self.url = 'https://api-inference.huggingface.co/models/' + model 
        self.headers = {"Authorization": f"Bearer {apiKey}"}
        self.key = apiKey
        self.model = model

    def query(self, query: str) -> str:
        response = requests.post(self.url, headers=self.headers, json=query)
        return response.json() # [0]['generated_text']
    

"""Client for interacting with OpenAI models"""
class OpenAIChatModel(Model):
    def __init__(self, apiKey: str, model: str, systemPrompt: str = None):
        self.client = OpenAI(api_key=apiKey)
        self.model = model
        
        # If the user does not set a custom set of messages, use a default
        if(systemPrompt is None):
            self.systemPrompt ="""
            You are a helpful AI assistant. Answer the inputted question with only the letter of the correct response without further explanation. Your response should be one character.
            """
        else:
            self.systemPrompt = systemPrompt


    """Generates an answer to a user query"""
    def query(self, query: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.systemPrompt},
                {"role": "user", "content": query}
            ]
        )

        return completion.choices[0].message.content
    

# TODO: OpenAI Completion model?

    

class AzureOpenAIModel(Model):
    """Client for interacting with azure openai solutions"""
    pass




