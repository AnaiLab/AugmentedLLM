from openai import OpenAI
import requests
import re

# TODO: Have augmented model inherit from this as well

"""Generic model class. Not to be instantiated, use a child class instead"""
class Model(object):
    def __init__(self, model: str):
        self.model = model
    
    """Generates an answer to the MCQ, and returns 1 if the answer is correct, 0 if it is incorrect, and -1 if it does not follow the expected format"""
    def gradeMCQ(self, query: str, numAnswers: int, answer: str) -> int:
        result = self.query(query)

        # Construct the regex to accept 1 letter replies
        regex = '^[A-Za-z]\\.?$'

        if not (self.__checkString(result,regex)):
            return -1
        elif result[0].lower() == answer[0].lower():
            return 1
        else:
            return 0
    
    """Check a string against the supplied regex pattern"""
    def __checkString(self, s: str, regex: str) -> bool:
        r = re.compile(regex)
        return bool(r.fullmatch(s))
    


"""Client for interacting with HuggingFace Models via inference API"""
class HuggingfaceRemoteModel(Model):
    def __init__(self, apiKey: str, model: str):
        self.url = 'https://api-inference.huggingface.co/models/' + model 
        self.headers = {"Authorization": f"Bearer {apiKey}"}
        self.key = apiKey
        super().__init__(model)

    def query(self, payload) -> str:
        response = requests.post(self.url, headers=self.headers, json=payload)
        return response.json()[0]['generated_text']
    

"""Client for interacting with OpenAI models"""
class OpenAIChatModel(Model):
    def __init__(self, apiKey: str, model: str, systemPrompt: str = None):
        self.client = OpenAI(api_key=apiKey)
        super().__init__(model)
        
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




