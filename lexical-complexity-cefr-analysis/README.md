# Lexical Complexity and Language Proficiency: CEFR Level Analysis

## Overview
This project investigates lexical complexity indices across CEFR (Common European Framework of Reference) levels using advanced machine learning techniques. The analysis combines LFTK (Lexical Features Toolkit) for feature extraction with LASSO regression and Random Forest algorithms to identify the most predictive lexical complexity measures for different proficiency levels.

## Research Context
This code corresponds to the published book chapter:

**Bozdağ, F. Ü., & Kılımcı, A. (2024). Lexical Complexity and Language Proficiency: An Investigation of Indices Across CEFR Levels. In İ. H. Mirici & H. Ergül (Eds.), *Current Academic Reflections on English Language Teaching in an EFL Setting* (pp. [page numbers]). Cambridge Scholars Publishing.**

## Features
- **LFTK Integration**: Extracts 76+ lexical complexity features using the Lexical Features Toolkit
- **LASSO L1 Regularization**: Identifies the most predictive features for each CEFR level
- **Random Forest Analysis**: Provides feature importance rankings across proficiency levels
- **Cross-linguistic Analysis**: Compares lexical complexity patterns across different L1 backgrounds
- **Visualization**: Creates comprehensive plots and heatmaps for results interpretation

## Pipeline Overview

### 1. Feature Extraction (`01_feature_extraction.py`)
- Uses spaCy and LFTK to extract lexical complexity features
- Processes learner texts from EFCAMDAT corpus
- Generates comprehensive feature matrix with 76+ lexical indices

### 2. LASSO Alpha Optimization (`02_lasso_alpha_optimization.py`)
- Optimizes LASSO regularization parameters using cross-validation
- Finds optimal alpha values for each CEFR level
- Provides baseline for feature selection

### 3. LASSO L1 Analysis (`03_lasso_l1_analysis.py`)
- Applies LASSO L1 regularization with optimized parameters
- Identifies top 10 most predictive features per CEFR level
- Creates visualizations and standardized coefficient analysis

### 4. Random Forest Analysis (`04_random_forest_analysis.py`)
- Validates LASSO results using Random Forest algorithm
- Analyzes feature importance across proficiency levels
- Provides cross-linguistic comparison by L1 background

## Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## Usage

### Step 1: Feature Extraction
```bash
python 01_feature_extraction.py
```

### Step 2: LASSO Alpha Optimization
```bash
python 02_lasso_alpha_optimization.py
```

### Step 3: LASSO L1 Analysis
```bash
python 03_lasso_l1_analysis.py
```

### Step 4: Random Forest Analysis
```bash
python 04_random_forest_analysis.py
```

## Data Requirements
- **EFCAMDAT Corpus**: Available upon request at https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html
- **Input Format**: CSV with columns including 'text_corrected' and CEFR level information
- **CEFR Levels**: A1, A2, B1, B2, C1 proficiency classifications

## Output Files
- `extracted_lexical_features.csv`: Complete feature matrix
- `lasso_alpha_optimization_results.csv`: Optimized alpha values
- `lasso_results.csv`: LASSO L1 coefficients
- `standardized_lasso_results.csv`: Standardized coefficients
- `random_forest_l1_results.csv`: Feature importance rankings
- `lasso_l1_heatmap.png`: Visualization of top features
- `random_forest_l1_analysis.png`: Cross-linguistic comparison plot

## Key Findings
The analysis reveals that different lexical complexity indices are predictive of different CEFR levels:

- **A1 Level**: Basic lexical diversity measures (TTR variants)
- **A2 Level**: Word frequency and POS variation
- **B1-B2 Levels**: Advanced lexical diversity and syntactic complexity
- **C1 Level**: Sophisticated lexical measures and semantic complexity

## Dependencies
- **NLP Processing**: spacy, lftk, thinc
- **Machine Learning**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **GPU Support**: torch (for spaCy acceleration)

## Requirements
- GPU recommended for feature extraction
- Sufficient RAM for large corpus processing
- Python 3.8+

## Research Applications
This tool is designed for:
- Second language acquisition research
- CEFR level assessment and validation
- Lexical complexity analysis in learner corpora
- Cross-linguistic comparison studies
- Automated proficiency level prediction

## Data Availability
The EFCAMDAT corpus used in this study is available upon request from the Cambridge English Language Assessment Research Team at https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html

## Author
**Fatih Ünal Bozdağ** - fbozdag1989@gmail.com  
**Abdurrahman Kılımcı** - Co-author

Osmaniye Korkut Ata University, Turkey
