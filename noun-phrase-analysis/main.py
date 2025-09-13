#!/usr/bin/env python3
"""
Noun Phrase Analysis for Learner English
========================================

This script analyzes various types of noun phrase modifications in learner English
using spaCy for linguistic annotation. It extracts and analyzes different types
of noun modifiers across CEFR levels and native languages.

Author: Fatih Ünal Bozdağ
Email: fbozdag1989@gmail.com
"""

import spacy
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import re
from scipy.stats import zscore
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure pandas and plotting
pd.set_option('display.max_colwidth', None)
plt.show(block=True)
plt.interactive(False)
sns.set_theme(style="darkgrid")

# Load the English NLP model
nlp = spacy.load("en_core_web_trf")
torch.device("mps")
spacy.require_gpu()

def annotate_attributive_adjective(corpus_with_context):
    """
    Annotate attributive adjectives modifying nouns.
    
    Args:
        corpus_with_context: List of tuples containing (text, context_dict)
    
    Returns:
        List of dictionaries with annotation information
    """
    annotations = []
    for doc, context in nlp.pipe(corpus_with_context, as_tuples=True):
        for chunk in doc.noun_chunks:
            adjectives = []
            # Explore each token in the noun chunk
            for token in chunk:
                if token.dep_ == 'amod' and token.pos_ == 'ADJ':
                    # Include adjectives and any conjunction-connected adjectives
                    if token not in adjectives:
                        adjectives.append(token)
                        adjectives.extend(child for child in token.conjuncts 
                                        if child.pos_ == 'ADJ' and child not in adjectives)

            if adjectives:
                # Sort adjectives by their index in the document
                sorted_adjectives = sorted(adjectives, key=lambda adj: adj.i)
                # Join all adjectives for a single noun into one string
                modifier_text = ' and '.join(adj.text for adj in sorted_adjectives)
                annotations.append({
                    "modifier_text": modifier_text,
                    "noun_text": chunk.root.text,
                    "noun_modifier": "Attributive Adjective",
                    "modifier_position": "pre",
                    "type": "phrasal",
                    "sentence": chunk.sent.text,
                    "native_language": context.get('l1', 'unknown'),
                    "cefr": context.get('cefr', 'unknown')
                })
    return annotations

def annotate_premodifying_nouns(corpus_with_context):
    """
    Annotate premodifying nouns (compound nouns).
    
    Args:
        corpus_with_context: List of tuples containing (text, context_dict)
    
    Returns:
        List of dictionaries with annotation information
    """
    annotations = []
    for doc, context in nlp.pipe(corpus_with_context, as_tuples=True):
        for chunk in doc.noun_chunks:
            # Inspect each token within the noun chunk
            for token in chunk:
                # Check if the token is a noun modifying another noun (compound) within a noun chunk
                if token.dep_ == 'compound' and token.head.pos_ == 'NOUN' and token.head in chunk:
                    annotations.append({
                        "modifier_text": token.text,
                        "noun_text": token.head.text,
                        "noun_modifier": "Premodifying Noun",
                        "modifier_position": "pre",
                        "type": "phrasal",
                        "sentence": token.sent.text,
                        "native_language": context.get('l1', 'unknown'),
                        "cefr": context.get('cefr', 'unknown')
                    })
    return annotations

def annotate_relative_clauses(corpus_with_context):
    """
    Annotate relative clauses modifying nouns.
    
    Args:
        corpus_with_context: List of tuples containing (text, context_dict)
    
    Returns:
        List of dictionaries with annotation information
    """
    annotations = []
    for doc, context in nlp.pipe(corpus_with_context, as_tuples=True):
        for token in doc:
            # Check if the token is a verb that is the root of a relative clause
            if token.dep_ == 'relcl' and token.head.pos_ == 'NOUN':
                # Build the relative clause text
                clause_text = ' '.join([tok.text_with_ws for tok in token.subtree]).strip()
                annotations.append({
                    "modifier_text": clause_text,
                    "noun_text": token.head.text,
                    "noun_modifier": "Relative Clause",
                    "modifier_position": "post",
                    "type": "clausal",
                    "sentence": token.sent.text,
                    "native_language": context.get('l1', 'unknown'),
                    "cefr": context.get('cefr', 'unknown')
                })
    return annotations

def annotate_ing_clauses(corpus_with_context):
    """
    Annotate -ing clauses modifying nouns.
    
    Args:
        corpus_with_context: List of tuples containing (text, context_dict)
    
    Returns:
        List of dictionaries with annotation information
    """
    annotations = []
    for doc, context in nlp.pipe(corpus_with_context, as_tuples=True):
        for token in doc:
            # Check for gerund verbs acting as adjectival clauses
            if token.pos_ == 'VERB' and token.dep_ == 'acl' and token.tag_ == 'VBG':
                # Build the -ing clause text
                clause_text = ' '.join([tok.text_with_ws for tok in token.subtree]).strip()
                annotations.append({
                    "modifier_text": clause_text,
                    "noun_text": token.head.text,
                    "noun_modifier": "-ing Clause",
                    "modifier_position": "post",
                    "type": "clausal",
                    "sentence": token.sent.text,
                    "native_language": context.get('l1', 'unknown'),
                    "cefr": context.get('cefr', 'unknown')
                })
    return annotations

def annotate_ed_clauses(corpus_with_context):
    """
    Annotate -ed clauses modifying nouns.
    
    Args:
        corpus_with_context: List of tuples containing (text, context_dict)
    
    Returns:
        List of dictionaries with annotation information
    """
    annotations = []
    for doc, context in nlp.pipe(corpus_with_context, as_tuples=True):
        for token in doc:
            # Check for past participle verbs acting as adjectival clauses
            if token.pos_ == 'VERB' and token.dep_ == 'acl' and token.tag_ == 'VBN':
                # Build the -ed clause text
                clause_text = ' '.join([tok.text_with_ws for tok in token.subtree]).strip()
                annotations.append({
                    "modifier_text": clause_text,
                    "noun_text": token.head.text,
                    "noun_modifier": "-ed Clause",
                    "modifier_position": "post",
                    "type": "clausal",
                    "sentence": token.sent.text,
                    "native_language": context.get('l1', 'unknown'),
                    "cefr": context.get('cefr', 'unknown')
                })
    return annotations

def annotate_prepositional_phrases_of(corpus_with_context):
    """
    Annotate prepositional phrases with 'of' modifying nouns.
    
    Args:
        corpus_with_context: List of tuples containing (text, context_dict)
    
    Returns:
        List of dictionaries with annotation information
    """
    annotations = []
    for doc, context in nlp.pipe(corpus_with_context, as_tuples=True):
        for token in doc:
            # Check for the preposition 'of' linking to a noun
            if token.text.lower() == 'of' and token.dep_ == 'prep':
                # The head of the preposition should be a noun and the object should also be a noun
                if token.head.pos_ == 'NOUN' and any(child.pos_ == 'NOUN' for child in token.children):
                    # Build the prepositional phrase text
                    phrase_text = 'of ' + ' '.join(child.text_with_ws for child in token.children 
                                                 if child.dep_ != 'punct').strip()
                    annotations.append({
                        "modifier_text": phrase_text,
                        "noun_text": token.head.text,
                        "noun_modifier": "Prepositional Phrase (of)",
                        "modifier_position": "post",
                        "type": "phrasal",
                        "sentence": token.sent.text,
                        "native_language": context.get('l1', 'unknown'),
                        "cefr": context.get('cefr', 'unknown')
                    })
    return annotations

def annotate_prepositional_phrases_other(corpus_with_context):
    """
    Annotate other prepositional phrases modifying nouns.
    
    Args:
        corpus_with_context: List of tuples containing (text, context_dict)
    
    Returns:
        List of dictionaries with annotation information
    """
    annotations = []
    for doc, context in nlp.pipe(corpus_with_context, as_tuples=True):
        for token in doc:
            # Check if the token is a preposition and not 'of'
            if token.dep_ == 'prep' and token.lemma_ != 'of':
                # Look for noun objects of the preposition
                pobj = next((child for child in token.children 
                           if child.dep_ == 'pobj' and child.pos_ == 'NOUN'), None)
                if pobj:
                    # Ensure the preposition directly follows a noun (check token's head is a noun)
                    if token.head.pos_ == 'NOUN':
                        phrase_text = token.text + ' ' + pobj.text
                        annotations.append({
                            "modifier_text": phrase_text,
                            "noun_text": token.head.text,
                            "noun_modifier": "Prepositional Phrase (other)",
                            "modifier_position": "post",
                            "type": "phrasal",
                            "sentence": token.sent.text,
                            "native_language": context.get('l1', 'unknown'),
                            "cefr": context.get('cefr', 'unknown')
                        })
    return annotations

def load_corpus_data(file_path):
    """
    Load and prepare corpus data for analysis.
    
    Args:
        file_path: Path to the CSV file containing the corpus data
    
    Returns:
        List of tuples containing (text, context_dict)
    """
    logger.info(f"Loading corpus data from {file_path}")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns (customize based on your data structure)
    columns_to_drop = ['topic_id', 'cefr_numeric', 'topic_id_original_categorical', 
                      'topic_id_original', 'topic_id_categorical', 'topic', 
                      'secondary_topic', 'topic_to_keep', 'time', 'text',
                      'writing_id', 'level', 'unit', 'text_number_per_learner_in_task',
                      'date', 'grade', 'wordcount', 'mtld']
    
    # Only drop columns that exist in the dataframe
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        df.drop(existing_columns_to_drop, axis=1, inplace=True)
    
    # Prepare corpus with context
    corpus = []
    for _, row in df.iterrows():
        context = {
            'l1': row.get('l1', 'unknown'),
            'cefr': row.get('cefr', 'unknown'),
            'nationality': row.get('nationality', 'unknown')
        }
        corpus.append((row.get('text_corrected', ''), context))
    
    logger.info(f"Loaded {len(corpus)} texts for analysis")
    return corpus

def analyze_noun_phrases(corpus):
    """
    Perform comprehensive noun phrase analysis on the corpus.
    
    Args:
        corpus: List of tuples containing (text, context_dict)
    
    Returns:
        DataFrame containing all annotations
    """
    logger.info("Starting noun phrase analysis...")
    
    # Run all annotation functions
    attributive_adjective = annotate_attributive_adjective(corpus)
    premodifying_nouns = annotate_premodifying_nouns(corpus)
    relative_clauses = annotate_relative_clauses(corpus)
    ing_clauses = annotate_ing_clauses(corpus)
    ed_clauses = annotate_ed_clauses(corpus)
    prepositional_phrases_of = annotate_prepositional_phrases_of(corpus)
    prepositional_phrases_other = annotate_prepositional_phrases_other(corpus)
    
    # Combine all annotations
    all_annotations = (attributive_adjective + premodifying_nouns + relative_clauses +
                      ing_clauses + ed_clauses + prepositional_phrases_of + 
                      prepositional_phrases_other)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_annotations)
    
    logger.info(f"Total annotations: {len(df)}")
    logger.info(f"Annotation types: {df['noun_modifier'].value_counts().to_dict()}")
    
    return df

def create_visualizations(df, output_dir="results"):
    """
    Create visualizations for noun phrase analysis results.
    
    Args:
        df: DataFrame containing annotation results
        output_dir: Directory to save visualization files
    """
    logger.info("Creating visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Analysis 1: Modifier distribution by CEFR level
    modifier_distribution = df.groupby(['noun_modifier', 'modifier_position', 'type', 'cefr']).size().unstack(fill_value=0).reset_index()
    
    # Apply Z-score normalization across each CEFR level column
    cefr_columns = modifier_distribution.columns[3:]
    for col in cefr_columns:
        modifier_distribution[col + '_z'] = zscore(modifier_distribution[col])
    
    # Melt the DataFrame for easier plotting
    data_melted = modifier_distribution.melt(
        id_vars=['noun_modifier', 'modifier_position', 'type'], 
        value_vars=[col + '_z' for col in cefr_columns], 
        var_name='CEFR_Level', 
        value_name='Z_Score'
    )
    
    # Create bar plot of Z-scores
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Z_Score', y='noun_modifier', hue='CEFR_Level', data=data_melted)
    plt.title('Z-Scores of Noun Modifier Usage by CEFR Level')
    plt.xlabel('Z-Score')
    plt.ylabel('Noun Modifier')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/noun_modifier_cefr_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analysis 2: Relative frequencies by CEFR level
    count_data = df.groupby(['noun_modifier', 'cefr']).size().reset_index(name='Count')
    total_counts_per_cefr = count_data.groupby('cefr')['Count'].sum()
    count_data['Relative_Frequency'] = count_data.apply(
        lambda row: row['Count'] / total_counts_per_cefr[row['cefr']], axis=1)
    
    # Create relative frequency plot
    plt.figure(figsize=(12, 8))
    g = sns.catplot(x='Relative_Frequency', y='noun_modifier', hue='cefr', 
                    col="cefr", data=count_data, capsize=.2, palette="YlGnBu_d", 
                    errorbar="se", kind="point", height=6, aspect=.75)
    g.fig.suptitle('Relative Frequencies of Noun Modifiers by CEFR Level', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/noun_modifier_relative_frequencies.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analysis 3: Native language analysis
    modifier_distribution_nl = df.groupby(['noun_modifier', 'modifier_position', 'type', 'native_language']).size().unstack(fill_value=0).reset_index()
    
    language_columns = modifier_distribution_nl.columns[3:]
    for col in language_columns:
        modifier_distribution_nl[col + '_z'] = zscore(modifier_distribution_nl[col])
    
    z_score_columns = [col for col in modifier_distribution_nl.columns if col.endswith('_z')]
    data_melted_nl = modifier_distribution_nl.melt(
        id_vars=['noun_modifier', 'modifier_position', 'type'], 
        value_vars=z_score_columns, 
        var_name='First_Languages', 
        value_name='Z_Score'
    )
    
    # Create native language analysis plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Z_Score', y='noun_modifier', hue='First_Languages', data=data_melted_nl)
    plt.title('Z-Scores of Noun Modifier Usage by First Languages')
    plt.xlabel('Z-Score')
    plt.ylabel('Noun Modifier')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/noun_modifier_l1_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Visualizations saved to {output_dir}/")

def main():
    """Main function to run the noun phrase analysis."""
    logger.info("Starting Noun Phrase Analysis for Learner English")
    
    # Load corpus data
    corpus = load_corpus_data('sample_data.csv')  # Replace with your data file
    
    # Perform analysis
    results_df = analyze_noun_phrases(corpus)
    
    # Save results
    results_df.to_csv('noun_phrase_results.csv', index=False)
    logger.info("Results saved to noun_phrase_results.csv")
    
    # Create visualizations
    create_visualizations(results_df)
    
    # Display summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total noun phrase modifications found: {len(results_df)}")
    print(f"Number of unique modifier types: {results_df['noun_modifier'].nunique()}")
    print(f"CEFR levels analyzed: {sorted(results_df['cefr'].unique())}")
    print(f"Native languages analyzed: {sorted(results_df['native_language'].unique())}")
    
    print("\n=== MODIFIER TYPE DISTRIBUTION ===")
    print(results_df['noun_modifier'].value_counts())
    
    logger.info("Analysis completed successfully!")

if __name__ == "__main__":
    main()
