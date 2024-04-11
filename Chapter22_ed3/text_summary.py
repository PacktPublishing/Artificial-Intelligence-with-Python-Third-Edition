from openai import OpenAI
client = OpenAI()

def summarize_text(text):
    prompt = f"Summarize the following text:\n\n{text}"
    print(prompt)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    summary = response.choices[0].message.content
    return summary


# Example text to summarize
text = """ 
The Transformer architecture has been widely adopted in natural language processing tasks.
"""

summary = summarize_text(text)
print("Summary:", summary)
