import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
from textblob import TextBlob
import PyPDF2
import matplotlib.pyplot as plt
import seaborn as sns
import string
import spacy
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

# Load spaCy model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def visualize_word2vec(model):
    words = list(model.wv.index_to_key)
    word_vectors = [model.wv[word] for word in words]
    word_vectors = np.array(word_vectors)
    perplexity = min(30, len(word_vectors) - 1)
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    reduced_vectors = tsne.fit_transform(word_vectors)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], c='b', marker='o', alpha=0.6)
    for i, word in enumerate(words):
        ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], word,
                color='orange', fontsize=10, alpha=0.7)
    ax.set_title('Word2Vec Word Embeddings Visualization (3D)')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    plt.tight_layout()
    plt.savefig('./Graphs/word2vec_3d_visualization.png')
    return fig

def train_word2vec_model(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    model = Word2Vec([filtered_words], min_count=1, vector_size=100, window=5, sg=0)
    return model

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''.join(page.extract_text() for page in reader.pages)
    return text, reader.pages

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    filtered_words = [word for word in filtered_words if word not in string.punctuation]
    corrected_text = TextBlob(" ".join(filtered_words)).correct()
    corrected_words = word_tokenize(str(corrected_text))
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in corrected_words]
    return " ".join(stemmed_words)

def ngram_analysis(text, n=2):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    n_grams = ngrams(filtered_words, n)
    return Counter(n_grams)

def sentiment_analysis(text):
    total_length = len(text)
    sections = {
        "Beginning": text[:total_length // 3],
        "Middle": text[total_length // 3: 2 * total_length // 3],
        "End": text[2 * total_length // 3:]
    }
    sentiments = {section: TextBlob(content).sentiment.polarity for section, content in sections.items()}
    return sentiments

def plot_combined_sentiment_and_ngram(sentiment_data, ngram_freq):
    sections = [section for page in sentiment_data for section in page.keys()]
    sentiment_scores = [score for page in sentiment_data for score in page.values()]
    ngram, counts = zip(*ngram_freq.most_common(10))
    ngram_labels = [' '.join(gram) for gram in ngram]
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=sections, y=sentiment_scores, marker="o", label="Sentiment Scores", color="b")
    for i, ngram_label in enumerate(ngram_labels):
        plt.text(i, sentiment_scores[i] + 0.05, ngram_label, fontsize=10, ha='center', va='bottom', color="orange")
    plt.title("Combined Sentiment Scores and N-gram Annotations")
    plt.xlabel("Story Sections")
    plt.ylabel("Sentiment Score")
    plt.ylim(-1, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./Graphs/combined_sentiment_ngram.png')
    return plt.gcf()

# Streamlit App
st.title("PDF Story Analysis")
st.write("Upload a PDF file to analyze sentiment and generate Word2Vec visualization.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    os.makedirs('./Graphs', exist_ok=True)
    text, pages = extract_text_from_pdf(uploaded_file)
    sentiment_data = []
    all_ngram_freq = Counter()

    for page in pages:
        page_text = page.extract_text()
        preprocessed_text = preprocess_text(page_text)
        ngram_freq = ngram_analysis(preprocessed_text, n=2)
        all_ngram_freq.update(ngram_freq)
        sentiments = sentiment_analysis(page_text)
        sentiment_data.append(sentiments)
        model = train_word2vec_model(page_text)
        word2vec_fig = visualize_word2vec(model)
        st.pyplot(word2vec_fig)

    combined_fig = plot_combined_sentiment_and_ngram(sentiment_data, all_ngram_freq)
    st.pyplot(combined_fig)
    st.success("Analysis Complete. Visualizations Generated!")
