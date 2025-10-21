import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

# Test sentence
doc = nlp("My car engine is making noise.")

# Print recognized entities
print("Entities found:", [ent.text for ent in doc.ents])
