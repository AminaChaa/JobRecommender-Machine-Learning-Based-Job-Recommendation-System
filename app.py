import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import pdfplumber  # Using pdfplumber for PDF parsing
import numpy as np

# Load your job postings data
df = pd.read_csv('data5.csv')  # Your job postings dataset

# Assuming you have a trained KNN model and a TF-IDF vectorizer
knn_model = NearestNeighbors(n_neighbors=5, n_jobs=-1)  # Your KNN model
tfidf_vectorizer = TfidfVectorizer(max_features=20, ngram_range=(1, 3), min_df=1)

# Pre-process the job descriptions
jd_tfidf = tfidf_vectorizer.fit_transform(df['processed_description']).toarray()  # Use 'processed_description'
knn_model.fit(jd_tfidf)

# Streamlit UI
st.title("Job Recommendation System")

# Upload CV
uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])
if uploaded_file is not None:
    # Parse the CV to extract skills and other details using pdfplumber
    with pdfplumber.open(uploaded_file) as pdf:
        resume_text = ""
        for page in pdf.pages:
            resume_text += page.extract_text()

    # Here, you would typically extract skills from the resume_text
    # For this example, we'll simply treat the entire text as skills
    skills_text = resume_text
    
    # Vectorize the CV
    cv_vector = tfidf_vectorizer.transform([skills_text]).toarray()  # Transform the CV into the same space as job descriptions
    
    # Get nearest job postings
    distances, indices = knn_model.kneighbors(cv_vector)
    
    # Display results
    st.subheader("Top 5 Job Matches")
    for i in range(5):
        job_index = indices[0][i]
        job_title = df.iloc[job_index]['Job Title']
        job_description = df.iloc[job_index]['Job Description']
        match_distance = distances[0][i]
        
        st.write(f"**Job Title**: {job_title}")
        st.write(f"**Job Description**: {job_description}")
        st.write(f"**Match Distance**: {match_distance:.2f}")
        st.write("---")

# Run the app using the command:
# streamlit run app.py
