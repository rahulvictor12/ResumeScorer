# ResumeScorer

ResumeScorer is a Python-based application designed to automate the evaluation of resumes. By extracting relevant information from resumes and comparing it against predefined criteria, the tool assigns a score to each resume, aiding recruiters in the screening process.

## Features

- **Resume Parsing**: Utilizes natural language processing techniques to extract key information such as skills, experience, and education from resumes.
- **Scoring Mechanism**: Compares extracted information against predefined criteria to assign a relevance score to each resume.
- **Data Extraction Modules**: Modular design for extracting data, allowing easy customization and extension.
- **Helper Methods**: Utility functions to support data processing and scoring operations.

## Project Structure

The repository is organized as follows:

- `.idea/`: IDE-specific settings and configurations.
- `Contracts/`: Defines interfaces and contracts used across the application.
- `DataExtractors/`: Modules responsible for extracting data from resumes.
- `DependencyInjection/`: Manages dependencies and their injections throughout the application.
- `HelperMethods/`: Contains utility functions to support various operations.
- `Models/`: Data models representing the structure of extracted information.
- `data/`: Directory intended for storing sample resumes and related data files.
- `RequirementInstaller.ipynb`: Jupyter Notebook for installing required dependencies.
- `ResumeScreener.ipynb`: Jupyter Notebook demonstrating the resume screening process.
- `ResumeScreener.py`: Main script to run the resume screening application.
- `app.py`: Entry point for the application.
- `requirements.txt`: Lists all Python dependencies required to run the application.

## Installation

To set up the ResumeScorer application on your local machine, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rahulvictor12/ResumeScorer.git
   
2. **Navigate to the Project Directory:**
cd ResumeScorer

3. **Create a Virtual Environment (optional but recommended):**
python3 -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'

4. **Install Dependencies:**
pip install -r requirements.txt

## Usage
After installation, you can use the ResumeScorer application as follows:

Import and Install neccesary Dependencies.
Run the Application:  python app.py

Upload the Job Description and Resume You wish to score.

This will process the resume in the ResumeScreener.py and output scores based on the predefined criteria.

## Contributing
Contributions to ResumeScorer are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.

## Acknowledgments
We appreciate the contributions of all developers and maintainers of the open-source libraries used in this project.

This `README.md` provides a comprehensive overview of the ResumeScorer project, including its features, structure, installation instructions, usage guidelines, contribution steps, license information, and acknowledgments.
::contentReference[oaicite:0]{index=0}
