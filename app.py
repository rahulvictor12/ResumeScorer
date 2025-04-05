import streamlit as st
from ResumeScreener import extract_resume_text
from DependencyInjection.container import Container
from HelperMethods.utils import set_background, inject_styles
import time

# Load container & model
container = Container()
model = container.ml_models()
sbert_model = model.load_sbert_model()

# Streamlit UI
st.set_page_config(page_title="Resume and JD Matching", layout="centered")

# === Set Background and Style ===
set_background("data/dot-particles-flowing-waves-3d-260nw-2450001013.webp")
inject_styles()

# Set the title
st.markdown("<div class='title'>Resume Screener</div>", unsafe_allow_html=True)

jd_text = st.text_area("Input Job Description", height=150)
resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=['pdf', 'doc', 'docx'])

left, center, right = st.columns([1, 2, 1])
placeholder = st.empty()  # Placeholder to render the dynamic button

with center:
    button_ph = st.empty()
    if button_ph.button("Analyze Resume Fit", use_container_width=True):
        if not jd_text or not resume_file:
            st.warning("Please upload both the Job Description and Resume.")
        else:
            # Replace the button with a "Processing..." message
            button_ph.markdown("<div style='text-align:center;'>🔄 <b>Processing...</b></div>", unsafe_allow_html=True)
            time.sleep(1)  # Simulate loading

            resume_text = extract_resume_text(resume_file)
            if "Error" in resume_text:
                st.error(resume_text)
            else:
                match_percent = model.calculate_matching_score(jd_text, resume_text, sbert_model)

                st.markdown(f"Matching Score: `{match_percent:.2f}%`")
                st.progress(min(match_percent / 100, 1.0))