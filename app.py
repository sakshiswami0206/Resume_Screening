import os
import pickle
import gdown
import streamlit as st
import re
import fitz  # from PyMuPDF

# Google Drive file IDs (ensure these files are publicly accessible)
clf_id = "1mZblfiK49Wvg2q1VljpryoUbsR75R9XP"
tfidf_id = "1r2739sjp1l3n28rwEqvlSxfr4aCv2Q-z"

clf_path = "clf.pkl"
tfidf_path = "tfidf.pkl"

# Function to download from Google Drive only if file doesn't exist
def download_if_needed(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        st.write(f"Downloading model file from: {url}")
        gdown.download(url, output_path, quiet=False)
    else:
        st.write(f"Found cached model: {output_path}")

# Function to verify if a file is a valid pickle
def is_valid_pickle(file_path):
    try:
        with open(file_path, "rb") as f:
            pickle.load(f)
        return True
    except Exception as e:
        st.error(f"Error loading model file: {file_path}\n{e}")
        return False

# Download model files
download_if_needed(clf_id, clf_path)
download_if_needed(tfidf_id, tfidf_path)

# Validate pickle files
if not is_valid_pickle(clf_path) or not is_valid_pickle(tfidf_path):
    st.stop()

# Load models
with open(clf_path, "rb") as f:
    clf = pickle.load(f)

with open(tfidf_path, "rb") as f:
    tfidf = pickle.load(f)

# Label mapping
id_to_label = {
    6: 'Data Science', 12: 'HR', 0: 'Advocate', 1: 'Arts', 24: 'Web Designing',
    16: 'Mechanical Engineer', 22: 'Sales', 14: 'Health and fitness',
    5: 'Civil Engineer', 15: 'Java Developer', 4: 'Business Analyst',
    21: 'SAP Developer', 2: 'Automation Testing', 11: 'Electrical Engineering',
    18: 'Operations Manager', 20: 'Python Developer', 8: 'DevOps Engineer',
    17: 'Network Security Engineer', 19: 'PMO', 7: 'Database',
    13: 'Hadoop', 10: 'ETL Developer', 9: 'DotNet Developer',
    3: 'Blockchain', 23: 'Testing'
}

# Resume text cleaning
def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# PDF text extractor
def extract_text_from_pdf(file):
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Main Streamlit app
def main():
    st.title("📄 Resume Classification App")

    uploaded_file = st.file_uploader("Upload a resume (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8", errors="ignore")

        if not text.strip():
            st.warning("No content found in the uploaded file.")
            return

        cleaned = clean_text(text)
        vector = tfidf.transform([cleaned])
        prediction = clf.predict(vector)[0]
        category = id_to_label.get(prediction, "Unknown")

        st.success(f"Predicted Category: **{category}**")

        if st.checkbox("Show Resume Text"):
            st.text_area("Extracted Resume Text", text, height=300)

if __name__ == "__main__":
    main()
