# Topic Modeling and Lexical Representation

## Overview
This project combines BERTopic for automatic topic modeling with Word2Vec for lexical representation analysis. It provides a comprehensive pipeline for discovering topics in learner English corpora and analyzing semantic relationships between words within specific topics.

## Features
- **BERTopic Integration**: Advanced topic modeling using transformer embeddings
- **ChatGPT Topic Representation**: Optional fine-tuning of topic labels using OpenAI's ChatGPT
- **Word2Vec Analysis**: Semantic relationship analysis for topic-specific vocabulary
- **Entropy Analysis**: Measures topic coherence and distribution
- **Visualization**: Interactive plots showing topic frequencies and semantic networks
- **Cross-linguistic Analysis**: Compares topic usage across different native languages

## Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## Usage
1. Prepare your data file with text and metadata:
```python
df = pd.read_csv("path/to/your/learner_corpus.csv")
```

2. Configure OpenAI API (optional):
```python
openai.organization = "your_OPENAI_Organization" 
openai.api_key = "your_OPENAI_API_Key"
```

3. Run the complete analysis:
```bash
python main.py
```

## Pipeline Overview

### 1. Text Preprocessing
- Removes document identifiers and special characters
- Lemmatizes text using spaCy
- Removes stop words
- Prepares documents for topic modeling

### 2. Topic Modeling with BERTopic
- **Embeddings**: Uses SentenceTransformer for document embeddings
- **Dimensionality Reduction**: UMAP for efficient clustering
- **Clustering**: HDBSCAN for automatic topic discovery
- **Topic Representation**: TF-IDF with optional ChatGPT enhancement
- **Entropy Calculation**: Measures topic coherence

### 3. Word2Vec Analysis
- Trains Word2Vec models on topic-specific texts
- Analyzes semantic relationships between education-related terms
- Creates semantic networks showing word associations

### 4. Visualization
- Topic frequency plots with entropy scores
- Semantic network visualizations
- Comparative analysis across languages

## Output
The analysis generates:
- **Topic Information**: Topic frequencies, labels, and entropy scores
- **Semantic Networks**: Word association networks for target vocabulary
- **Visualizations**: Bar plots and network graphs
- **Word Similarities**: Cosine similarity scores for related terms

## Configuration Parameters

### BERTopic Settings
```python
umap_model = UMAP(n_neighbors=20, n_components=5, min_dist=0.0, metric='cosine', random_state=123)
hdbscan_model = HDBSCAN(min_cluster_size=30, metric='euclidean', cluster_selection_method='eom')
```

### Word2Vec Settings
```python
model = Word2Vec(vector_size=500, window=10, min_count=100, workers=10, sg=0, epochs=50)
```

## Dependencies
- **Topic Modeling**: bertopic, umap-learn, hdbscan, scikit-learn
- **NLP Processing**: spacy, sentence-transformers, keybert
- **Word Embeddings**: gensim
- **Visualization**: matplotlib, seaborn, networkx
- **AI Integration**: openai

## Target Vocabulary Analysis
The script focuses on education-related vocabulary:
- academic, education, university, graduate, college
- Analyzes semantic relationships and contextual usage
- Provides similarity scores for related terms

## Data Format
Input CSV should contain:
- **text_field**: The main text content
- **Native_Language**: Native language of the writer
- Additional metadata columns as needed

## Reproducibility
- Uses fixed random seeds for consistent results
- Documents all parameter settings
- Provides detailed configuration options

## Performance Optimization
- GPU acceleration for transformer models
- Efficient batch processing
- Memory-optimized text preprocessing
- Parallel processing where possible

## Research Applications
This tool is designed for:
- Corpus linguistics research
- Second language acquisition studies
- Topic modeling in learner corpora
- Semantic analysis of educational discourse
- Cross-linguistic comparison studies

## Author
Fatih Ünal Bozdağ - fbozdag1989@gmail.com
