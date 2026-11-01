import nltk
import spacy
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')

text = """Natural language processing (NLP) is a subfield of linguistics,
computer science, and artificial intelligence. NLP techniques are widely used
in sentiment analysis, machine translation, chatbots, and information retrieval."""

# NLTK Pipeline
tokens_nltk = word_tokenize(text)
stop_words = set(stopwords.words('english'))
filtered_nltk = [t for t in tokens_nltk if t.lower() not in stop_words and t.isalpha()]
stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered_nltk]
lemmatizer = WordNetLemmatizer()
lemmatized_nltk = [lemmatizer.lemmatize(w,pos='v') for w in filtered_nltk]
pos_tags_nltk = nltk.pos_tag(filtered_nltk)

# spaCy Pipeline
doc = nlp(text)
filtered_spacy = [token.text for token in doc if not token.is_stop and token.is_alpha]
lemmas_spacy = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
pos_spacy = [(token.text, token.pos_) for token in doc if not token.is_stop and
token.is_alpha]

min_len = min(
    len(filtered_nltk),
    len(stemmed),
    len(lemmatized_nltk),
    len(lemmas_spacy),
    len(pos_tags_nltk),
    len(pos_spacy)
)

comparison = pd.DataFrame({
    'Word': filtered_nltk[:min_len],
    'NLTK_Stem': stemmed[:min_len],
    'NLTK_Lemma': lemmatized_nltk[:min_len],
    'spaCy_Lemma': lemmas_spacy[:min_len],
    'NLTK_POS': [t[1] for t in pos_tags_nltk[:min_len]],
    'spaCy_POS': [t[1] for t in pos_spacy[:min_len]]
})
print(comparison.to_string(index=False))