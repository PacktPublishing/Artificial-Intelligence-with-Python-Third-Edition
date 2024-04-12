from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

template = """You are a chatbot having a conversation with a human.

{chat_history}
Human: {human_input}
Chatbot:"""

# Initialize the LLM
llm = OpenAI()

# Set up the chatbot using ChatChain
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")
chatbot = LLMChain(llm=llm,
                   prompt=prompt,
                   memory=memory)

# Example conversation
response = chatbot.predict(human_input='Hello there! I am Mike. How are you?')
print(response)

response = chatbot.predict(human_input='Do you remember what is my name?')
print(response)


