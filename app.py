from flask import Flask, render_template, request
import pickle
import re
import string
from transformers import pipeline

app = Flask(__name__)

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------
# Load GPT-2 (lighter model)
# -----------------------------
summarizer = pipeline("text-generation", model="distilgpt2")

# -----------------------------
# Suspicious words list
# -----------------------------
suspicious_words = [
    "shocking", "unbelievable", "secret",
    "viral", "exposed", "miracle", "breaking"
]

# -----------------------------
# Text Cleaning (SAME AS TRAINING)
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    # -----------------------------
    # CLEAN INPUT BEFORE PREDICTION
    # -----------------------------
    cleaned_news = clean_text(news)

    vect_news = vectorizer.transform([cleaned_news])
    prediction = model.predict(vect_news)
    probability = model.predict_proba(vect_news)

    confidence = round(max(probability[0]) * 100, 2)

    result = "🟢 Real News" if prediction[0] == 1 else "🔴 Fake News"

    # -----------------------------
    # Generate Summary
    # -----------------------------
    try:
        summary_input = news[:500]
        summary = summarizer(
            "Summarize this news in simple words: " + summary_input,
            max_length=120,
            num_return_sequences=1
        )
        summary_output = summary[0]['generated_text']
    except:
        summary_output = "Summary could not be generated."

    # -----------------------------
    # Highlight suspicious words (case insensitive)
    # -----------------------------
    highlighted_news = news
    for word in suspicious_words:
        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
        highlighted_news = pattern.sub(
            f"<span style='color:red; font-weight:bold;'>{word}</span>",
            highlighted_news
        )

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        summary=summary_output,
        highlighted_news=highlighted_news
    )


if __name__ == "__main__":
    app.run(debug=True)