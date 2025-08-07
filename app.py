import streamlit as st
import pickle
import re
import nltk
import os
import gdown
import fitz  # PyMuPDF

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# File IDs for Google Drive
clf_drive_id = "1mZblfiK49Wvg2q1VljpryoUbsR75R9XP"
tfidf_drive_id = "YOUR_TFIDF_PKL_FILE_ID"  # ← Replace with tfidf.pkl file ID

# Local paths
clf_path = "clf.pkl"
tfidf_path = "tfidf.pkl"

# Download clf.pkl if it's not already present
if not os.path.exists(clf_path):
    gdown.download(f"https://drive.google.com/uc?id={clf_drive_id}", clf_path, quiet=False)

# Download tfidf.pkl if it's not already present
if not os.path.exists(tfidf_path):
    gdown.download(f"https://drive.google.com/uc?id={tfidf_drive_id}", tfidf_path, quiet=False)

# Load the model and vectorizer using pickle
with open(clf_path, 'rb') as f:
    clf = pickle.load(f)

with open(tfidf_path, 'rb') as f:
    tfidf = pickle.load(f)

# Your existing text cleaning and classification logic follows…

# Label mapping
id_to_label = {
    6: 'Data Science',
    12: 'HR',
    0: 'Advocate',
    1: 'Arts',
    24: 'Web Designing',
    16: 'Mechanical Engineer',
    22: 'Sales',
    14: 'Health and fitness',
    5: 'Civil Engineer',
    15: 'Java Developer',
    4: 'Business Analyst',
    21: 'SAP Developer',
    2: 'Automation Testing',
    11: 'Electrical Engineering',
    18: 'Operations Manager',
    20: 'Python Developer',
    8: 'DevOps Engineer',
    17: 'Network Security Engineer',
    19: 'PMO',
    7: 'Database',
    13: 'Hadoop',
    10: 'ETL Developer',
    9: 'DotNet Developer',
    3: 'Blockchain',
    23: 'Testing'
}

# Cleaning function
def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# PDF extractor
def extract_text_from_pdf(file):
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Streamlit app
def main():
    st.title("📄 Resume Classification App")

    uploaded_file = st.file_uploader("Upload a resume (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file:
        # Extract content
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8", errors="ignore")

        if not text.strip():
            st.warning("No content found in the uploaded file.")
            return

        # Clean, transform, predict
        cleaned = clean_text(text)
        vector = tfidf.transform([cleaned])
        prediction = clf.predict(vector)[0]

        category = id_to_label.get(prediction, "Unknown")
        st.success(f"Predicted Category: **{category}**")

        if st.checkbox("Show Resume Text"):
            st.text_area("Extracted Resume Text", text, height=300)

if __name__ == "__main__":
    main()
