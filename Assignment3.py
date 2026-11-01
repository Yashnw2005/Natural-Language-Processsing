import numpy as np
import matplotlib.pyplot as plt 
from gensim.models import Word2Vec
import gensim.downloader as api
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1. Custom Corpus
sentences = [
    ['natural', 'language', 'processing', 'is', 'a', 'field',
     'of', 'artificial', 'intelligence'],

    ['machine', 'learning', 'algorithms', 'learn', 'patterns',
     'from', 'data'],

    ['deep', 'learning', 'uses', 'neural', 'networks',
     'with', 'multiple', 'layers'],

    ['word', 'embeddings', 'capture', 'semantic', 'meaning',
     'of', 'words'],

    ['transformers', 'use', 'attention', 'mechanism', 'for',
     'sequence', 'modeling'],

    ['bert', 'is', 'a', 'pre', 'trained', 'transformer',
     'model', 'for', 'nlp']
]

# 2. Train Word2Vec Model
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    epochs=100,
    seed=42
)
print("Vocabulary Size:", len(model.wv))
print("Vector Size:", model.vector_size)

# 3. Semantic Similarity
print("\nMost similar to 'neural':")

for word, score in model.wv.most_similar('neural', topn=5):
    print(word, ":", round(score, 4))


# 4. Cosine Similarity
print("\nCosine Similarity:")

print(
    "sim(neural, networks) =",
    round(model.wv.similarity('neural', 'networks'), 4)
)

print(
    "sim(learning, deep) =",
    round(model.wv.similarity('learning', 'deep'), 4)
)


# 5. Select Words for Visualization
viz_words = [
    'neural',
    'networks',
    'deep',
    'learning',
    'language',
    'nlp',
    'transformer'
]

viz_words = [
    w for w in viz_words
    if w in model.wv
]

vectors = np.array([
    model.wv[w]
    for w in viz_words
])


# 6. PCA Visualization
pca = PCA(n_components=2)

coords_pca = pca.fit_transform(vectors)

plt.figure(figsize=(8, 6))

plt.scatter(
    coords_pca[:, 0],
    coords_pca[:, 1],
    s=80
)

for i, w in enumerate(viz_words):

    plt.annotate(
        w,
        (
            coords_pca[i, 0] + 0.01,
            coords_pca[i, 1] + 0.01
        ),
        fontsize=9
    )

plt.title('Word Embeddings - PCA Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, alpha=0.3)

plt.savefig(
    'embeddings_pca.png',
    dpi=100
)
plt.show()


# 7. t-SNE Visualization
tsne = TSNE(
    n_components=2,
    perplexity=3,
    random_state=42,
    init='pca',
    learning_rate='auto'
)

coords_tsne = tsne.fit_transform(vectors)

plt.figure(figsize=(8, 6))

plt.scatter(
    coords_tsne[:, 0],
    coords_tsne[:, 1],
    s=80
)

for i, w in enumerate(viz_words):

    plt.annotate(
        w,
        (
            coords_tsne[i, 0] + 0.01,
            coords_tsne[i, 1] + 0.01
        ),
        fontsize=9
    )

plt.title('Word Embeddings - t-SNE Projection')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True, alpha=0.3)

plt.savefig(
    'embeddings_tsne.png',
    dpi=100
)
plt.show()


# 8. Load Pre-trained GloVe
print("\nLoading pre-trained GloVe vectors...")

glove = api.load('glove-wiki-gigaword-100')

print("GloVe loaded successfully.")
print("GloVe Vocabulary Size:", len(glove))
print("GloVe Vector Size:", glove.vector_size)


# 9. GloVe Semantic Similarity
print("\nGloVe - Most similar to 'neural':")
for word, score in glove.most_similar('neural', topn=5):

    print(
        word,
        ":",
        round(score, 4)
    )


# 10. GloVe Analogy
# king - man + woman = queen-
print("\nGloVe Analogy:")
print("king - man + woman ≈ queen")

result = glove.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=5
)

for word, score in result:

    print(
        word,
        ":",
        round(score, 4)
    )


# 11. Final Summary
print("\n----------------------------------------")
print("Assignment Completed Successfully")
print("----------------------------------------")

print("Word2Vec Vocabulary Size:", len(model.wv))
print("Word2Vec Vector Size:", model.vector_size)

print(
    "Neural-Networks Similarity:",
    round(
        model.wv.similarity(
            'neural',
            'networks'
        ),
        4
    )
)
print("\nGenerated files:")
print("1. embeddings_pca.png")
print("2. embeddings_tsne.png")