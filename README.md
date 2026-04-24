# Language Detection ML

A machine learning application that detects the language of given text using Naive Bayes classification.

## Project Structure

```
Language Dectection ML/
├── app.py              # Streamlit web application
├── model.py            # Model training script
├── language.csv        # Training dataset
├── requirements.txt    # Python dependencies
├── model.pkl           # Trained model (generated)
└── vectorizer.pkl      # Fitted vectorizer (generated)
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train the model:**
```bash
python model.py
```

3. **Run the web app:**
```bash
streamlit run app.py
```

## Usage

1. After running `streamlit run app.py`, a browser window will open
2. Enter text in any language in the text area
3. Click "Detect Language" to see the predicted language

## Supported Languages

The model is trained on text samples from:
- English
- Estonian
- Swedish
- Thai
- Tamil
- Dutch
- Japanese
- Turkish
- Latin
- Urdu
- Indonesian
- Portuguese
- French
- Chinese
- Korean

## How It Works

1. **Data Loading:** Reads language-labeled text from `language.csv`
2. **Vectorization:** Uses `CountVectorizer` to convert text to numerical features
3. **Training:** Trains a `MultinomialNB` (Naive Bayes) classifier
4. **Prediction:** Transforms input text using the fitted vectorizer and predicts language

## Requirements

- pandas
- numpy
- scikit-learn
- streamlit