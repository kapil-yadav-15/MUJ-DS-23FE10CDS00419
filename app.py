
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# -------------------------------
# 1. Page configuration
# -------------------------------
st.set_page_config(
    page_title="Language Detection",
    page_icon="🌍",
    layout="centered"
)

DATA_PATH = Path(__file__).parent / "language.csv"


# -------------------------------
# 2. Load dataset and train model
# -------------------------------
@st.cache_resource(show_spinner="Preparing your language model...")
def load_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.name}. "
            "Place language.csv beside app.py."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"Text", "language"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "The dataset must contain 'Text' and 'language' columns."
        )

    df = df.dropna(subset=["Text", "language"]).copy()

    texts = df["Text"].astype(str)
    labels = df["language"].astype(str)

    # Convert text into numerical word-count features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    # Train the language classification model
    model = MultinomialNB()
    model.fit(X, labels)

    return vectorizer, model, len(df)


# -------------------------------
# 3. Interface design
# -------------------------------
st.title("🌍 Language Detection App")

st.write(
    "Enter a sentence or paragraph and let the machine "
    "learning model predict its language."
)

st.info(
    "This application uses CountVectorizer and "
    "Multinomial Naive Bayes."
)

# Load the trained model
try:
    vectorizer, model, total_records = load_model()

except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# -------------------------------
# 4. Display project information
# -------------------------------
col1, col2 = st.columns(2)

col1.metric("Languages Supported", len(model.classes_))
col2.metric("Training Records", f"{total_records:,}")

st.divider()


# -------------------------------
# 5. Take user input
# -------------------------------
st.subheader("Enter your text")

user_text = st.text_area(
    "Text to analyze",
    placeholder="Example: Bonjour, comment allez-vous?",
    height=150
)

st.caption(
    "Tip: A complete sentence usually gives the model "
    "more information than a single word."
)


# -------------------------------
# 6. Predict the language
# -------------------------------
if st.button(
    "🔎 Detect Language",
    type="primary",
    use_container_width=True
):

    if not user_text.strip():
        st.warning("Please enter some text first.")

    else:
        # Convert the input using the same vectorizer
        input_vector = vectorizer.transform([user_text])

        # Predict language
        prediction = model.predict(input_vector)[0]

        # Get model scores for the available classes
        probabilities = model.predict_proba(input_vector)[0]

        st.divider()
        st.subheader("📊 Prediction Result")

        st.success(f"Predicted Language: {prediction}")

        st.caption(
            "Prediction is based on patterns learned from "
            "your training dataset."
        )

        # Show the top three predictions
        top_indices = probabilities.argsort()[::-1][:3]

        results = pd.DataFrame({
            "Language": model.classes_[top_indices],
            "Model score (%)": (
                probabilities[top_indices] * 100
            ).round(2)
        })

        st.subheader("Top Language Predictions")
        st.dataframe(results, use_container_width=True, hide_index=True)

        st.bar_chart(
            results.set_index("Language")["Model score (%)"]
        )

        st.caption(
            "Model scores are not necessarily calibrated probabilities "
            "or measures of actual prediction accuracy."
        )


# -------------------------------
# 7. Footer
# -------------------------------
st.divider()

st.caption(
    "NLP Project | Python | Scikit-learn | Streamlit"
)
