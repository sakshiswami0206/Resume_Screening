import streamlit as st
import pickle
import re
import nltk
import fitz  # PyMuPDF for PDF text extraction

# Download NLTK resources (only needed once)
nltk.download('stopwords')
nltk.download('punkt')


# Load pre-trained model and vectorizer
import os
import gdown

# Define file paths and Google Drive file IDs
clf_file_path = "clf.pkl"
tfidf_file_path = "tfidf.pkl"

clf_drive_id = "1r2739sjp1l3n28rwEqvlSxfr4aCv2Q-z"  # Replace with actual clf.pkl ID
tfidf_drive_id = "1abcdefghijklmno...xyz"           # Replace with actual tfidf.pkl ID

# Download clf.pkl if not exists
if not os.path.exists(clf_file_path):
    clf_url = f"https://drive.google.com/uc?id={clf_drive_id}"
    gdown.download(clf_url, clf_file_path, quiet=False)

# Download tfidf.pkl if not exists
if not os.path.exists(tfidf_file_path):
    tfidf_url = f"https://drive.google.com/uc?id={tfidf_drive_id}"
    gdown.download(tfidf_url, tfidf_file_path, quiet=False)

# Mapping IDs to category names
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


# Text cleaning function
def clean_res(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape("""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~"""), " ", text)
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(RT|cc)\b", " ", text, flags=re.IGNORECASE)
    return text.strip()


# PDF text extractor
def extract_text_from_pdf(uploaded_file):
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            return " ".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"Failed to extract PDF text: {e}")
        return ""


# Main app
def main():
    st.title("📄 Resume Screening App")
    st.write("Upload a resume (PDF or TXT) to classify its category.")

    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "txt"])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            try:
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
            except:
                resume_text = resume_bytes.decode('latin-1')

        if not resume_text.strip():
            st.warning("Couldn't extract any text from the uploaded file.")
            return

        # Clean and transform resume text
        cleaned_text = clean_res(resume_text)
        vectorized_text = tfidf.transform([cleaned_text])
        prediction_id = clf.predict(vectorized_text)[0]

        category = id_to_label.get(prediction_id, "Unknown")

        st.success(f"🎯 **Predicted Category:** {category}")

        if st.checkbox("Show extracted resume text"):
            st.subheader("📄 Extracted Resume Content")
            st.text_area("Resume Text", resume_text, height=300)


if __name__ == "__main__":
    main()
