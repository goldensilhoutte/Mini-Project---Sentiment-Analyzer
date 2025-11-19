from transformers import pipeline
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (only first time)
nltk.download('stopwords')

print("Loading sentiment analysis model...")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
print("Model loaded successfully!\n")

print("Welcome to the Python Sentiment Analyzer!")
print("Type 'quit' to exit.\n")

max_length = 512  # Max tokens allowed for DistilBERT

while True:
    text = input("Enter a sentence: ")

    if text.lower() == "quit":
        break

    if not text.strip():
        print("Please enter some text!\n")
        continue

    try:
        result = sentiment_analyzer(
            text,
            truncation=True,
            max_length=max_length
        )[0]

        print(f"Sentiment: {result['label']} | Confidence: {result['score']:.2f}\n")

    except Exception as e:
        print("Error analyzing sentiment:", e)
        print()
