import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer , WordNetLemmatizer
from nltk import pos_tag

#download the required sources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

#input
text = "Machine learning algorithms learn patterns from historical data"
print("Original Text:")
print(text)

#Tokenization
tokens=word_tokenize(text.lower())
print("\n1. Tokens:")
print(tokens)

#Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in tokens if word.isalpha() and word not in stop_words]
print("\n2. After Stopword Removal:")
print(filtered_words)

#POS Tagging
pos_tags = pos_tag(filtered_words)
print("\n3. POS Tags:")
print(pos_tags)

#Stemming
stemmer = PorterStemmer()
stemmed_words = [
    stemmer.stem(word)
    for word in filtered_words
]
print("\n4. After Stemming:")
print(stemmed_words)

#Lemmatization
lemmatizer = WordNetLemmatizer()

lemmatized_words = [
    lemmatizer.lemmatize(word)
    for word in filtered_words
]

print("\n5. After Lemmatization:")
print(lemmatized_words)