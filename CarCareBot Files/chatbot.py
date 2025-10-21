import pandas as pd
import os
import datetime
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

CSV_FILENAME = "enhanced_motor_vehicle_repair_towing_dataset.csv"
MODEL_FILE = "diagnostic_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    raise Exception("SpaCy model not found. Run: python -m spacy download en_core_web_sm")

# Load dataset
def load_dataset():
    if not os.path.exists(CSV_FILENAME):
        print(f"Warning: CSV file not found: {CSV_FILENAME}")
        # Create fallback dataset
        data = {
            "service_description": [
                "engine making noise",
                "battery dead",
                "flat tire replacement",
                "oil change required",
                "brake pads worn",
                "towing needed"
            ],
            "service_type": [
                "engine repair",
                "battery replacement",
                "tire service",
                "oil change",
                "brake service",
                "towing"
            ],
            "vehicle_type": ["car", "car", "car", "car", "car", "car"],
            "make_and_model": ["generic", "generic", "generic", "generic", "generic", "generic"]
        }
        return pd.DataFrame(data)

    df = pd.read_csv(CSV_FILENAME, low_memory=False)
    df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]
    for col in ['service_description', 'service_type', 'vehicle_type', 'make_and_model']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('')
    return df

# Search dataset
def search_columns(df, keyword, limit=3):
    kw = keyword.lower()
    search_cols = ['service_description', 'service_type', 'make_and_model']
    search_cols = [c for c in search_cols if c in df.columns]
    matches = df[df[search_cols].apply(lambda r: r.astype(str).str.lower().str.contains(kw).any(), axis=1)]
    if matches.empty:
        return f"No records found for '{keyword}'."
    results = []
    for _, row in matches.head(limit).iterrows():
        results.append(", ".join([f"{col}: {row[col]}" for col in search_cols]))
    return "\n\n---\n\n".join(results)

# Train model
def train_diagnostic_model():
    df = load_dataset()
    X = df['service_description']
    y = df['service_type'].fillna('general_service')
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(X_vec, y)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

# Predict diagnosis
def predict_diagnosis(query):
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        train_diagnostic_model()
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    X_vec = vectorizer.transform([query])
    return model.predict(X_vec)[0]

# Maintenance guide
def generate_maintenance_guide(issue):
    guides = {
        "battery replacement": "Check battery terminals, clean corrosion, and replace battery every 3–4 years.",
        "engine repair": "Change oil regularly, monitor engine noise, and visit a mechanic if warning lights appear.",
        "towing": "Use proper tow hooks, avoid exceeding weight limits, and ensure brake lights work.",
        "brake service": "Inspect brake pads regularly and replace them if they are thin or squealing.",
        "oil change": "Replace oil every 5000 km or as recommended by the manufacturer.",
        "tire service": "Check tire pressure monthly and replace worn or punctured tires.",
        "general_service": "Perform regular vehicle check-ups for best performance."
    }
    return guides.get(issue.lower(), "Perform regular vehicle check-ups for best performance.")

# Main logic
def get_answer(query):
    try:
        df = load_dataset()
        search_result = search_columns(df, query)
        diagnosis = predict_diagnosis(query)
        guide = generate_maintenance_guide(diagnosis)
        response = f"{search_result}\n\nPredicted Diagnosis: {diagnosis}\n\nMaintenance Guide:\n{guide}"
        return response
    except Exception as e:
        return f"Error in get_answer: {e}"

# Log chat interactions
def log_interaction(user_input, ai_response):
    log_file = "carcarebot_training_data.csv"
    header = ["timestamp", "user_input", "ai_response"]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.exists(log_file)
    import csv
    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([timestamp, user_input, ai_response])
