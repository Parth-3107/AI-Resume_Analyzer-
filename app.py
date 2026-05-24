import PyPDF2
import streamlit as st 
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text 

def extract_skills(text):
    doc = nlp(text)
    skills = set()

    # FIXED: Changed from "for token.pos_ in [...]" to iterate over tokens
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            clean_token = token.text.strip().lower().rstrip('.,:;!?')
            if clean_token:
                skills.add(clean_token)

    for chunk in doc.noun_chunks:
        clean_chunk = chunk.text.strip().lower().rstrip('.,:;!?')
        if " " in clean_chunk: 
            skills.add(clean_chunk)

    return list(skills)

# FIXED: Function name was misspelled (claculate_ats_score -> calculate_ats_score)
def calculate_ats_score(resume_text, job_description):
    if not resume_text or not job_description:
        return 0.0

    tfidf_vec = TfidfVectorizer(stop_words='english')
    tfidf_mat = tfidf_vec.fit_transform([resume_text, job_description])

    similarity_score = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0]
    return round(similarity_score * 100, 2)

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("AI Resume Analyzer")
st.markdown("Upload your Resume and paste a job description to check how well you match")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload your Resume")
    uploaded_file = st.file_uploader("Choose a PDF File", type="pdf")
    resume_text = ""
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            resume_text = extract_text_from_pdf(uploaded_file)
        st.success(" PDF text extracted successfully!")
        
        with st.expander("Preview Extracted Text"):
            st.write(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)

with col2:
    st.subheader("2. Paste Job Description")
    job_description = st.text_area("Paste the job description here...", height=300)
    
    # FIXED: Indentation error - the if/else block was not properly indented
    if st.button("Analyze Resume", type="primary"):
        if not uploaded_file:
            st.warning("Please upload a resume PDF first.")
        elif not job_description:
            st.warning("Please paste a job description.")
        else:
            st.divider() 
            st.header(" Analysis Results")
            
            with st.spinner("Extracting skills..."):
                skills_found = extract_skills(resume_text)
                
            with st.spinner("Calculating ATS Match Score..."):
                ats_score = calculate_ats_score(resume_text, job_description)
            
            # Display the results in a clean layout
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric(label="ATS Match Score", value=f"{ats_score}%")
                # Add a simple interpretation
                if ats_score > 70:
                    st.success("Strong Match! Your resume aligns well.")
                elif ats_score > 40:
                    st.info("Moderate Match. Consider tailoring your resume further.")
                else:
                    st.warning("Low Match. Your resume may need significant adjustments.")
                    
            with res_col2:
                st.subheader(" Extracted Keywords/Skills")
                if skills_found:
                    st.write(", ".join(f"`{skill}`" for skill in sorted(skills_found[:30])))
                    st.caption("Showing up to 30 unique terms extracted as potential skills.")
                else:
                    st.write("No specific skill keywords could be extracted.")