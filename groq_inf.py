import pandas as pd
from groq import Groq
import config

client = Groq(
    api_key=config.GROQ_TOKEN
)

df = pd.read_csv("final_data_balanced.csv")

models = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "gemma2-9b-it",
]


def get_sentiment(text, sentiment_model):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Classify the sentiment of this text as 1 for positive and 0 for negative. Just give the sentiment score i.e. 0 or 1: '{text}'",
            }
        ],
        model=sentiment_model,
    )

    sentiment = response.choices[0].message.content.strip()
    print("TEXT - ", text)
    print("SENTIMENT MODEL- ", sentiment_model)
    print("SENTIMENT - ", sentiment)
    print("*" * 50)
    return sentiment


for model in models:
    df[model] = df["text"].apply(lambda text: get_sentiment(text, model))


# Save the updated DataFrame
df.to_csv("output_with_sentiment_balanced.csv", index=False)
