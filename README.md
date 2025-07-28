# Resume Analyzer & Matcher (LangChain + Google Generative AI + Streamlit)
### Note: This is the beta version (v1) and lacks file upload feature. Please see v2 with improved file upload feature. 
---

## 1. Project Objective

This project provides an interactive tool for analyzing and matching resumes against job requirements using Google Generative AI (Gemini) and LangChain.  
It extracts text from resumes and job descriptions, uses an LLM to analyze suitability, and stores results in a Chroma vector store for future retrieval.

---

## 2. Main Libraries

- **Streamlit**: Interactive web UI.
- **LangChain**: Document loaders, prompt templates, chains.
- **Google Generative AI (Gemini)**: LLM for resume/job analysis and embeddings.
- **Chroma**: Vector store for storing/retrieving analyses.
- **python-dotenv**: Loads API keys from `.env`.
- **re**: Regex for extracting scores.

---

## 3. Top Functions and Variables

| Name                        | Purpose                                                                                   |
|-----------------------------|-------------------------------------------------------------------------------------------|
| `extract_text_from_resume`  | Loads and extracts text from PDF, DOCX, or TXT files using LangChain loaders.             |
| `split_text`                | Splits long analysis text into manageable chunks for vector storage.                      |
| `store_resume_analysis`     | Stores the resume analysis in the Chroma vector store.                                    |
| `extract_suitability_score` | Uses regex to extract the "Suitability Score" from the LLM's analysis output.             |
| `GOOGLE_API_KEY`            | API key for Google Generative AI, loaded from `.env`.                                     |
| `vectorstore`               | Chroma vector store instance for storing/retrieving analyses.                             |
| `chain`                     | LangChain pipeline for generating the resume-job analysis using Gemini.                   |
| `resume_text`, `job_requirements` | Variables holding extracted text from uploaded files.                                |

---

## 4. How to Run the Project

1. **Clone the repository**  
   ```
   git clone <your-repo-url>
   cd <project-folder>
   ```

2. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

3. **Set up your `.env` file**  
   - Create a `.env` file in the project root.
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     ```

4. **Run the Streamlit app**  
   ```
   streamlit run Resume1.py
   ```

5. **Use the app**  
   - Store your resume in the folder where .py file is saved (PDF, DOCX, or TXT).
   - Replace Python and Java in line 89 (below)and add your job description (PDF, DOCX, or TXT).
      - job_requirements = "Python and Java"
   - View the analysis and suitability score.

---

## 5. File Structure

```
Resume_Project1/
├── second/
│   ├── Resume1.py
│   ├── requirements.txt
│   └── .env
├── chroma_store/
│   └── ... (vector store data)
```

---

## 6. Additional Notes

- **Supported File Types**: PDF, DOCX, TXT for both resume and job requirements.
- **Vector Store**: Analyses are stored for future retrieval and can be extended for search or recommendations.
- **Error Handling**: The app will display errors for unsupported formats or missing API keys.
- **Extensibility**: You can add more features, such as candidate ranking, feedback, or multi-resume comparison.

---


## 7. Contact

For questions or contributions, please open an issue or pull request on GitHub.

