import pandas as pd
import openai
import config

client = openai.OpenAI(
    api_key=config.OPENAI_API_KEY
)

df = pd.read_csv("output_with_sentiment_balanced.csv")
df = df[:200]  # Limiting the data to the first 200 rows

models = [
    "gpt-3.5-turbo",
    "gpt-4o",
]


def get_sentiment(text, sentiment_model):
    response = client.chat.completions.create(
        model=sentiment_model,  # Using the specified OpenAI model
        messages=[
            {
                "role": "user",
                "content": f"Classify the sentiment of this text as 1 for positive and 0 for negative. Just give the sentiment score (0 or 1), nothing else: '{text}'",
            }
        ],
    )

    sentiment = response.choices[0].message.content.strip()
    print("TEXT - ", text)
    print("SENTIMENT MODEL - ", sentiment_model)
    print("SENTIMENT - ", sentiment)
    print("*" * 50)
    return sentiment


for model in models:
    df[model] = df["text"].apply(lambda text: get_sentiment(text, model))

# Save the updated DataFrame
df.to_csv("output_with_sentiment_balanced_with_gpt.csv", index=False)
