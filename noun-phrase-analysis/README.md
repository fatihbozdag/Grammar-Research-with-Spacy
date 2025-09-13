# Noun Phrase Analysis in L2 Writing

## Overview
This project analyzes noun phrase complexity in learner English writing using spaCy for linguistic annotation and statistical analysis. The analysis focuses on different types of noun modifiers across CEFR levels and native languages to understand the development of syntactic complexity in second language acquisition.

## Research Context
This code corresponds to the published paper:

**Bozdağ FÜ, Mo J, Morris G (2025) A corpus-based analysis of noun modifiers in L2 writing: The respective impact of L2 proficiency and L1 background. PLoS ONE 20(3): e0320092. https://doi.org/10.1371/journal.pone.0320092**

📖 **Full article available at:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0320092

### Citation Details
- **Title:** A corpus-based analysis of noun modifiers in L2 writing: The respective impact of L2 proficiency and L1 background
- **Authors:** Fatih Ünal Bozdağ, Junhua Mo, Gareth Morris
- **Journal:** PLoS ONE
- **Volume:** 20(3)
- **DOI:** https://doi.org/10.1371/journal.pone.0320092
- **Year:** 2025

## Features
- **Comprehensive Noun Modifier Analysis**: Extracts 7 different types of noun modifiers
- **CEFR Level Analysis**: Analyzes noun phrase complexity across proficiency levels
- **Cross-linguistic Comparison**: Compares patterns across different L1 backgrounds
- **Statistical Analysis**: Uses Z-score normalization and relative frequency analysis
- **Visualization**: Creates comprehensive plots for results interpretation

## Noun Modifier Types Analyzed

### Premodifiers
1. **Attributive Adjectives**: Adjectives modifying nouns (e.g., "beautiful car")
2. **Premodifying Nouns**: Compound nouns (e.g., "car door")

### Postmodifiers
3. **Relative Clauses**: Clauses modifying nouns (e.g., "the car that I bought")
4. **-ing Clauses**: Gerund clauses (e.g., "the man walking")
5. **-ed Clauses**: Past participle clauses (e.g., "the book written")
6. **Prepositional Phrases (of)**: "of" phrases (e.g., "the door of the car")
7. **Prepositional Phrases (other)**: Other prepositional phrases (e.g., "the book on the table")

## Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## Usage
```bash
python main.py
```

## Data Requirements
- **EFCAMDAT Corpus**: Available at https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html
- **Input Format**: CSV with columns including 'text_corrected', 'cefr', 'l1', 'nationality'
- **CEFR Levels**: A1, A2, B1, B2, C1 proficiency classifications

## Output Files
- `noun_phrase_results.csv`: Complete annotation results
- `results/noun_modifier_cefr_analysis.png`: CEFR level analysis visualization
- `results/noun_modifier_relative_frequencies.png`: Relative frequency analysis
- `results/noun_modifier_l1_analysis.png`: Native language analysis

## Key Findings
The analysis reveals that:

1. **L2 Proficiency Impact**: L2 proficiency has a significant impact on noun phrase complexity development
2. **L1 Background Influence**: L1 background shows observable but limited influence on noun modifier patterns
3. **Convergence Pattern**: As proficiency increases, learners tend to converge towards common grammatical competence
4. **Modifier Type Preferences**: Different CEFR levels show distinct patterns in modifier type usage

## Statistical Methods
- **Z-score Normalization**: Standardizes modifier usage across CEFR levels
- **Relative Frequency Analysis**: Calculates proportional usage within each proficiency level
- **Cross-tabulation**: Analyzes distribution patterns across multiple variables
- **Visualization**: Creates comparative plots for pattern identification

## Dependencies
- **spaCy**: Natural language processing and linguistic annotation
- **pandas/numpy**: Data manipulation and numerical analysis
- **matplotlib/seaborn**: Statistical visualization
- **scipy**: Statistical functions and Z-score calculation

## Requirements
- GPU recommended for spaCy processing
- Sufficient RAM for large corpus analysis
- Python 3.8+

## Research Applications
This tool is designed for:
- Second language acquisition research
- Noun phrase complexity analysis
- CEFR level assessment and validation
- Cross-linguistic comparison studies
- Academic writing development research

## Data Availability
The data presented in this study are accessible at https://github.com/fatihbozdag/Grammar-Research-with-Spacy/blob/main/noun-phrase-analysis

## Funding
This research was funded by the Humanities and Social Sciences Interdisciplinary Research Team of Soochow University (Grant No: 5033720623).

## Authors
**Fatih Ünal Bozdağ** - fbozdag1989@gmail.com  
Osmaniye Korkut Ata University, Turkey

**Junhua Mo** - sdjunhua@suda.edu.cn  
Soochow University, China

**Gareth Morris**  
University of Nottingham, Ningbo, China
