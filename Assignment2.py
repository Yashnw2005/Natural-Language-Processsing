import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample Documents
corpus = [
    "I love natural language processing",
    "I love machine learning and NLP",
    "Natural language processing is part of AI",
    "I love deep learning and neural networks"
]

# Tokenization
def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())

# Build Vocabulary
def build_vocab(corpus):
    vocab = set()
    for doc in corpus:
        vocab.update(tokenize(doc))
    return sorted(vocab)

# Bag of Words
def bow_vector(doc, vocab):
    counts = Counter(tokenize(doc))
    return [counts.get(word, 0) for word in vocab]

vocab = build_vocab(corpus)
bow_matrix = np.array([bow_vector(doc, vocab) for doc in corpus])

print("========== Vocabulary ==========")
print(vocab)

print("\n========== Bag of Words Matrix ==========")
print(bow_matrix)

# TF-IDF From Scratch
def compute_tfidf(corpus):
    tokens = [tokenize(doc) for doc in corpus]
    vocab = sorted(set(word for doc in tokens for word in doc))

    N = len(corpus)

    bow = np.array([
        [Counter(doc).get(word, 0) for word in vocab]
        for doc in tokens
    ])

    df = (bow > 0).sum(axis=0)
    idf = np.log((1 + N) / (1 + df)) + 1

    tf = bow / bow.sum(axis=1, keepdims=True)

    tfidf = tf * idf

    norm = np.linalg.norm(tfidf, axis=1, keepdims=True)

    tfidf = tfidf / norm

    return tfidf, vocab

tfidf_matrix, vocab = compute_tfidf(corpus)

print("\n========== TF-IDF Matrix ==========")
print(np.round(tfidf_matrix,3))

# Verify using CountVectorizer
print("\n========== CountVectorizer ==========")

cv = CountVectorizer()

X = cv.fit_transform(corpus)

print(cv.get_feature_names_out())
print(X.toarray())

# Verify using TfidfVectorizer
print("\n========== TfidfVectorizer ==========")

tv = TfidfVectorizer()

Y = tv.fit_transform(corpus)

print(tv.get_feature_names_out())
print(np.round(Y.toarray(),3))

# Cosine Similarity
print("\n========== Cosine Similarity ==========")

similarity = cosine_similarity(tfidf_matrix)

print(np.round(similarity,3))

# Top N Keywords
def top_keywords(tfidf_matrix, vocab, doc_index, n=5):
    scores = tfidf_matrix[doc_index]
    top = np.argsort(-scores)[:n]
    return [(vocab[i], round(scores[i],3)) for i in top]

print("\n========== Top Keywords ==========")

for i in range(len(corpus)):
    print(f"Document {i+1}:")
    print(top_keywords(tfidf_matrix, vocab, i))