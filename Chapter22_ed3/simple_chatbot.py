from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# Initialize the LLM
llm = OpenAI()

# Set up the chatbot using ChatChain
prompt_template = "Tell me a story about {something}."
prompt = PromptTemplate(
    input_variables=["something"], template=prompt_template
)

chatbot = LLMChain(llm=llm, prompt=prompt)

# Example conversation
response = chatbot('a cat')

print(response['text'])