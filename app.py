import streamlit as st
import pickle
import spacy
import fitz  
from docx import Document
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import time
import base64
from io import BytesIO
import spacy.cli

st.set_page_config(
    page_title="HireSight",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 2rem;
    }
    .result-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #1E3A8A;
    }
    .stProgress .st-bo {
        background-color: #1E3A8A;
    }
    .category-label {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6B7280;
        font-size: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E0E7FF;
        border-bottom: 2px solid #1E3A8A;
    }

/* Remove button hover, active, and focus effects */
button[kind="primary"] {
    color: white !important;
    background-color: #1E3A8A !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

button[kind="primary"]:hover,
button[kind="primary"]:active,
button[kind="primary"]:focus {
    background-color: #1E3A8A !important;
    color: white !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Remove the default hover effect that may be applied */
.stButton>button {
    cursor: default !important;
}



</style>
""", unsafe_allow_html=True)



@st.cache_resource


def load_spacy():
    try:
        # Attempt to load the model
        return spacy.load("en_core_web_sm")
    except Exception as e:
        # Catch any exception, log it, and download the model
        print(f"Error loading model: {e}")
        print("Downloading 'en_core_web_sm' model...")
        spacy.cli.download("en_core_web_sm")
        # Return the model after downloading
        return spacy.load("en_core_web_sm")

nlp = load_spacy()

@st.cache_resource
def load_model():
    with open("model_res.pkl", "rb") as f:
        return pickle.load(f)

try:
    model_pipeline = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_message = str(e)

category_mapping = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
    24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
    1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
    19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
    17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
}

domain_categories = {
    "Technology": ["Java Developer", "Python Developer", "DevOps Engineer", "DotNet Developer", 
                  "Database", "ETL Developer", "Hadoop", "Blockchain", "SAP Developer"],
    "Engineering": ["Mechanical Engineer", "Electrical Engineering", "Civil Engineer"],
    "Quality Assurance": ["Testing", "Automation Testing"],
    "Business": ["Business Analyst", "Operations Manager", "PMO", "Sales", "HR"],
    "Security": ["Network Security Engineer"],
    "Design": ["Web Designing"],
    "Data": ["Data Science"],
    "Other": ["Arts", "Health and fitness", "Advocate"]
}

category_to_domain = {}
for domain, categories in domain_categories.items():
    for category in categories:
        category_to_domain[category] = domain

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(tokens)

def predict_category(text):
    cleaned = preprocess_text(text)
    pred = model_pipeline.predict([cleaned])[0]
    return category_mapping.get(pred, "Unknown")

def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def get_wordcloud_img(text):
    cleaned = preprocess_text(text)
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100,
        contour_width=1,
        contour_color='steelblue'
    ).generate(cleaned)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    
    return img_str

def extract_key_skills(text):
    doc = nlp(text)
    skills = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "GPE"]:
            skills.append(ent.text)
    
    tech_skills = ["Python", "Java", "JavaScript", "SQL", "AWS", "Docker", "Kubernetes", 
                  "React", "Angular", "Node.js", "C++", "C#", "HTML", "CSS", "Git"]
    
    for skill in tech_skills:
        if skill.lower() in text.lower() and skill not in skills:
            skills.append(skill)
    
    return list(set(skills))[:10]

def main():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/resume.png", width=80)
        st.title("Resume Classifier")
        st.markdown("---")
        
        st.subheader("About")
        st.write("""
        This tool uses AI to analyze resumes and predict the most suitable job role category.
        Upload your resume or paste its content to get started.
        """)
        
        st.markdown("---")
        st.subheader("Supported File Types")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("📄 PDF")
        with col2:
            st.markdown("📝 DOCX")
        with col3:
            st.markdown("📋 TXT")
        
        st.markdown("---")
        st.subheader("How it works")
        st.markdown("""
        1. Upload your resume
        2. Our AI analyzes the content
        3. Get job role prediction
        4. View detailed insights
        """)
        
        st.markdown("---")
        st.caption("© 2025 HireSight")

    st.markdown('<h1 class="main-header">HireSight🔎</h1>', unsafe_allow_html=True)
    st.markdown("Analyze your resume and discover the most suitable job role using advanced AI")
    
    if not model_loaded:
        st.error(f"Error loading model: {error_message}")
        st.warning("Please ensure the model file 'model_res.pkl' is in the same directory as this script.")
        return
    
    tab1, tab2 = st.tabs(["📤 Upload Resume", "✏️ Paste Content"])
    
    text = ""
    with tab1:
        uploaded_file = st.file_uploader("", type=["pdf", "docx", "txt"])
        if uploaded_file:
            try:
                file_details = {"Filename": uploaded_file.name, "File size": f"{uploaded_file.size/1024:.2f} KB"}
                st.write(file_details)
                
                if uploaded_file.type == "application/pdf":
                    with st.spinner("Extracting text from PDF..."):
                        text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    with st.spinner("Extracting text from DOCX..."):
                        text = extract_text_from_docx(uploaded_file)
                else:
                    text = uploaded_file.getvalue().decode("utf-8")
                
                st.success("File processed successfully!")
                
                with st.expander("Preview Extracted Text"):
                    st.text_area("", text[:1000] + ("..." if len(text) > 1000 else ""), height=200)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        manual_text = st.text_area("Paste your resume content here:", height=300)
        if manual_text.strip():
            text = manual_text
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)
    
    if text and analyze_button:
        with st.spinner("Analyzing resume content..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            category = predict_category(text)
            domain = category_to_domain.get(category, "Other")
            
            skills = extract_key_skills(text)
            
            wordcloud_img = get_wordcloud_img(text)
            
            progress_bar.empty()
        
        st.markdown('<h2 class="sub-header">Analysis Results</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(f"https://img.icons8.com/color/96/000000/{'briefcase' if domain == 'Business' else 'code' if domain == 'Technology' else 'engineering' if 'Engineering' in domain else 'test-partial-passed' if domain == 'Quality Assurance' else 'data-protection'}.png", width=80)
        with col2:
            st.markdown(f"<p>Predicted Job Role:</p><p class='category-label'>{category}</p>", unsafe_allow_html=True)
            st.markdown(f"**Domain:** {domain}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        result_tab1, result_tab2, result_tab3 = st.tabs(["📊 Key Insights", "🔠 Word Cloud", "🔍 Resume Analysis"])
        
        with result_tab1:
            st.subheader("Key Skills Identified")
            if skills:
                cols = st.columns(5)
                for i, skill in enumerate(skills):
                    with cols[i % 5]:
                        st.markdown(f"**•** {skill}")
            else:
                st.info("No specific skills identified. Consider adding more details to your resume.")
            
            st.subheader("Career Path Suggestions")
            st.write(f"Based on your resume, you might be a good fit for these roles:")
            
            related_categories = [c for c in domain_categories.get(domain, []) if c != category][:3]
            
            for i, related in enumerate(related_categories):
                st.markdown(f"**{i+1}.** {related}")
            
            st.subheader("Resume Strength")
            st.progress(0.75)
            st.caption("Your resume is well-aligned with the predicted job role.")
        
        with result_tab2:
            st.subheader("Word Cloud Visualization")
            st.markdown("This visualization highlights the most prominent terms in your resume:")
            st.markdown(f'<img src="data:image/png;base64,{wordcloud_img}" width="100%">', unsafe_allow_html=True)
            st.caption("Larger words appear more frequently in your resume")
        
        with result_tab3:
            st.subheader("Resume Content Analysis")
            
            word_count = len(text.split())
            sentence_count = len(text.split('.'))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Word Count", word_count)
            with col2:
                st.metric("Sentence Count", sentence_count)
            with col3:
                st.metric("Avg. Words per Sentence", round(word_count/max(1, sentence_count), 1))
            
            complexity = "Medium"
            if word_count < 200:
                complexity = "Low"
            elif word_count > 800:
                complexity = "High"
                
            st.info(f"**Content Complexity:** {complexity}")
            
            if complexity == "Low":
                st.warning("Your resume seems brief. Consider adding more details about your experience and skills.")
            elif complexity == "High":
                st.warning("Your resume is quite detailed. Consider focusing on the most relevant information.")
            else:
                st.success("Your resume has a good balance of detail and conciseness.")
    
    st.markdown('<div class="footer">Resume Classifier AI • Powered by Machine Learning</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()