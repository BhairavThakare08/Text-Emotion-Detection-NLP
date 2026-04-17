# ================= IMPORTS =================
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import re
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

st.markdown("""
<style>

/* ===== Main Background ===== */
.stApp {
    background-color: #0B0F19;
}

/* ===== Text ===== */
h1, h2, h3, h4, h5, h6, p, label {
    color: #E6EDF3 !important;
}

/* ===== Input Box ===== */
textarea {
    background-color: #111827 !important;
    color: #E6EDF3 !important;
    border-radius: 10px !important;
    border: 1px solid #1F2937 !important;
}

/* Container of textarea */
div[data-baseweb="textarea"] {
    background-color: #111827 !important;
    border-radius: 10px !important;
}

/* ===== Submit Button ===== */

/* ===== FORCE BLACK SUBMIT BUTTON ===== */
div.stForm button {
    background-color: #020617 !important;
    color: #E6EDF3 !important;
    border-radius: 8px !important;
    border: 1px solid #1F2937 !important;
}

/* Hover */
div.stForm button:hover {
    background-color: #111827 !important;
    color: white !important;
}

/* Click */
div.stForm button:active {
    background-color: #000000 !important;
}

/* Focus fix */
div.stForm button:focus {
    outline: none !important;
    box-shadow: none !important;
    background-color: #020617 !important;
}

/* ===== Green Sections (like cards) ===== */
.stAlert {
    background: linear-gradient(90deg, #134E4A, #166534) !important;
    color: #E6EDF3 !important;
    border-radius: 10px;
}

/* ===== Chart background fix ===== */
.css-1kyxreq, .css-1v0mbdj {
    background-color: transparent !important;
}

/* Remove white blocks */
.block-container {
    background-color: transparent !important;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ================= LABEL MAP =================
label_map = {
    0: "joy",
    1: "sadness",
    2: "anger",
    3: "fear",
    4: "love",
    5: "surprise"
}

# ================= EMOJI =================
emotions_emoji_dict = {
    "joy": "😂",
    "sadness": "😔",
    "anger": "😠",
    "fear": "😨😱",
    "love": "❤️",
    "surprise": "😮"
}

# ================= CLEANING =================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ================= PREDICTION =================
def predict_emotions(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    return label_map[pred]

def get_prediction_proba(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    probs = model.predict_proba(vector)
    return probs


# ================= MAIN UI =================
def main():
    st.title("🧠 Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    # Input Form
    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    # ================= OUTPUT =================
    if submit_text:

        # ⭐ ADDED (INPUT VALIDATION)
        if raw_text.strip() == "":
            st.warning("⚠️ Please type something to detect emotion")
            return

        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        # LEFT SIDE
        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write(f"{prediction} {emoji_icon}")

            st.write(f"Confidence: {np.max(probability):.2f}")

        # RIGHT SIDE
        with col2:
            st.success("Prediction Probability")

            proba_df = pd.DataFrame(
                probability,
                columns=list(label_map.values())
            )

            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='emotions',
                y='probability',
                color='emotions'
            )

            st.altair_chart(fig, use_container_width=True)


# ================= RUN =================
if __name__ == '__main__':
    main()
