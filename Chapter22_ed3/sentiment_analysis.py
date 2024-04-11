from openai import OpenAI

client = OpenAI()

def analyze_sentiment(text):
    prompt = f"What is the sentiment of this text?\n\n{text}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    sentiment = response.choices[0].message.content

    return sentiment


# Example text to analyze

text = "I had an amazing experience with the product, and the customer service was excellent."

sentiment = analyze_sentiment(text)

print("Sentiment:", sentiment)