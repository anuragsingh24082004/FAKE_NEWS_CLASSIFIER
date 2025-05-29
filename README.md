# Fake News Classifier

This is a machine learning project that classifies news articles as either real or fake using a PassiveAggressiveClassifier and TF-IDF vectorization.

## Features
- Text cleaning and preprocessing
- TF-IDF vectorization
- PassiveAggressiveClassifier model
- Streamlit web interface
- Model and vectorizer persistence with pickle

## How to Run
1. Install dependencies:
```
pip install -r requirements.txt
```

2. Train and save the model:
```
python train.py
```

3. Run the Streamlit app:
```
streamlit run app/streamlit_app.py
```

## Project Structure
```
.
├── app/
│   └── streamlit_app.py
├── models/
│   ├── model.pkl
│   └── tfidf.pkl
├── data/
│   ├── True.csv
│   └── Fake.csv
├── train.py
├── requirements.txt
└── README.md
```