import streamlit as st
import fitz  # PyMuPDF
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Function to analyze resume and job description
def analyze_resume_and_job_description(resume_text, job_description):
    preprocessed_resume_text = preprocess_text(resume_text)
    preprocessed_job_description = preprocess_text(job_description)
    
    vectorizer = CountVectorizer().fit([preprocessed_resume_text, preprocessed_job_description])
    vectors = vectorizer.transform([preprocessed_resume_text, preprocessed_job_description])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    similarity_percentage = similarity * 100
    
    common_keywords = set(preprocessed_job_description.split()) & set(preprocessed_resume_text.split())
    missing_keywords = set(preprocessed_job_description.split()) - set(preprocessed_resume_text.split())
    
    return similarity_percentage, common_keywords, missing_keywords

# Function to analyze resume
def analyze_resume(resume_file, job_description):
    resume_text = extract_text_from_pdf(resume_file)
    similarity_percentage, common_keywords, missing_keywords = analyze_resume_and_job_description(resume_text, job_description)
    return similarity_percentage, common_keywords, missing_keywords

def main():
    st.title("Resume Analyzer")
    st.markdown("---")

    uploaded_resume = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])
    job_description = st.text_area("Enter Job Description", height=200)

    if uploaded_resume is not None and job_description:
        st.markdown("---")
        with st.spinner('Analyzing resume...'):
            similarity_percentage, common_keywords, missing_keywords = analyze_resume(uploaded_resume, job_description)
            st.success('Analysis complete!')
            
            st.subheader("Analysis Result:")
            st.write(f"**Similarity between the resume and the job description:** {similarity_percentage:.2f}%")
            
            st.markdown("**Common Keywords:**")
            if common_keywords:
                st.info(', '.join(common_keywords))
            else:
                st.info("*No common keywords found.*")
                
            st.markdown("**Missing Keywords in Resume:**")
            if missing_keywords:
                st.error(', '.join(missing_keywords))
            else:
                st.error("*No missing keywords found.*")
            
            if similarity_percentage < 50:
                st.error("Your resume may not be well aligned with the job description. Consider updating it to include more relevant keywords.")
            elif similarity_percentage < 80:
                st.warning("Your resume has a moderate alignment with the job description. Consider optimizing it further to improve the match.")
            else:
                st.success("Your resume is well aligned with the job description. Congratulations!")

if __name__ == "__main__":
    main()
