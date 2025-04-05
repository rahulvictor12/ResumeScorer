import warnings
import os
import pandas as pd
import spacy
warnings.filterwarnings("ignore")
from io import BytesIO

# Import your DI container and components
from DependencyInjection.container import Container

# Initialize the container
container = Container()

#  spaCy (better for resume parsing since it includes NER and POS out-of-the-box)
nlp = spacy.load('en_core_web_sm')

# Extract Text from PDF file/ Path.
def extract_resume_text(file):
    try:
        extracted_text = ''
        if isinstance(file, str):
            if not os.path.exists(file):
                raise FileNotFoundError('File not found')

            if file.split()[-1].endswith('.pdf'):
                extractor = container.pdf_extractor()
            elif file.split()[-1].endswith(('.doc', '.docx')):
                extractor = container.word_extractor()
            else:
                return "Error: Unsupported file format (only PDF/DOC/DOCX accepted)"

            extracted_text = extractor.extract_text_from_path(file)

        elif isinstance(file, BytesIO) or hasattr(file, "read"):
            file_name = getattr(file, "name").lower()
            if file_name.endswith('.pdf'):
                extractor = container.pdf_extractor()
            elif file_name.endswith(('.doc', '.docx')):
                extractor = container.word_extractor()
            else:
                return "Error: Unsupported file format (only PDF/DOC/DOCX accepted)"

            extracted_text = extractor.extract_text_from_pdf_file(file)

        if extracted_text.isspace():
            return "Error: No Text could be Extracted from file"

        return extracted_text

    except Exception as e:
        return f'Error: {e}'


def process_text_with_spacy(text):
    doc = nlp(text)

    # Preprocessing of Text:
    #Tokenize the Text, Excluding Stop-words and Punctuations:
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    # Lemmatization:
    lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    # Named Entity Recognition
    entities = {ent.text: ent.label_ for ent in doc.ents}

    return {"tokens": tokens, "lemmas": lemmas, "entities": entities}

'''
# Clean and Read the Resume and Description Data from the CSV's
resumeDF = pd.read_csv('data/Resume.csv', nrows = 100)
jdDF = pd.read_csv('data/fake_job_postings.csv', nrows = 100)
jdDF['requirements'] = jdDF['requirements'].fillna('').astype(str)

# Extract the Resumes and JD's
resume_text = []
jd_text = []

for i, row in resumeDF.iterrows():
    text = row['Resume_str'] + '\n\n'
    resume_text.append(text)

for i , row in jdDF.iterrows():
    text = row['title'] + '\n\n' + row['description'] + '\n\n' + row['requirements']
    jd_text.append(text)

resume_processed = [process_text_with_spacy(text) for text in resume_text]
description_processed = [process_text_with_spacy(text) for text in jd_text]
'''