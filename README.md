# Citizen Feedback Sentiment Analysis System

This project analyzes citizen feedback about public services and classifies the sentiment into Positive, Negative, or Neutral. It features a complete pipeline from an ML training notebook to an interactive web dashboard.

## Project Structure
```text
citizen-feedback-sentiment-analysis/
│
├── backend/
│   ├── app/
│   │   └── sentiment_training.ipynb    # ML training and preprocessing notebook
│   │
│   └── model/
│       ├── sentiment_model.pkl         # Trained Logistic Regression model
│       ├── tfidf_vectorizer.pkl        # TF-IDF text vectorizer
│       └── data.csv                    # Dataset used for training
│
├── frontend/
│   └── app.py                          # Streamlit web dashboard
│
├── requirements.txt
└── README.md
```

## Requirements
See `requirements.txt`. Core libraries used include `pandas`, `numpy`, `scikit-learn`, `nltk`, `plotly`, and `streamlit`.

## How to run

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model (Optional if .pkl files already exist):**
   - Navigate to `backend/app/`
   - Run `sentiment_training.ipynb` using Jupyter Notebook or your preferred IDE.
   - This script will process the dataset and generate the `.pkl` models required by the frontend in `backend/model/`.

3. **Run the Streamlit web app:**
   - Navigate to the `frontend/` directory.
   - Execute the following command:
     ```bash
     streamlit run app.py
     ```
   - Streamlit will run a local web server (usually at `http://localhost:8501`).

## System Workflow Pipeline
Raw Feedback Dataset → TF-IDF Vectorization → Logistic Regression Training → Model Artifacts Saved → Streamlit Visual UI → User Predictions
