from openai import OpenAI

client=OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[{"role": "user",
             "content": "Describe a sunset as seen from a mountain top."}]
)

print(completion.choices[0].message.content)
