# Dative Alternation Analysis

## Overview
This project analyzes English dative constructions in academic writings of English EFL learners using spaCy for natural language processing. The analysis focuses on probabilistic patterns of dative alternation between double object constructions and prepositional dative constructions.

## Research Context
This code corresponds to the published paper:

**Bozdağ, F. Ünal. (2024). Probabilistic Analysis of English Dative Constructions in Academic Writings of English EFL Learners. *Theory and Practice of Second Language Acquisition*, *10*(1), 1–24. https://doi.org/10.31261/TAPSLA.13902**

📖 **Full article available at:** https://journals.us.edu.pl/index.php/TAPSLA/article/view/13902

## Features
- **Double Object Dative Analysis**: Extracts patterns like "give someone something"
- **Prepositional Dative Analysis**: Extracts patterns like "give something to someone"
- **Statistical Analysis**: Calculates length-based features and construction type distributions
- **Learner Corpus Analysis**: Processes ICLE (International Corpus of Learner English) data

## Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## Usage
1. Prepare your data files:
   - `metadata_with_text.csv`: Metadata file with document information
   - `text_only.csv`: Text data file

2. Update file paths in the `load_data()` function:
```python
meta = pd.read_csv('path/to/your/metadata_with_text.csv')
text = pd.read_csv('path/to/your/text_only.csv')
```

3. Run the analysis:
```bash
python main.py
```

## Output
The script generates a combined DataFrame with the following columns:
- `dative_sentences`: Full sentences containing dative constructions
- `native_language`: Native language of the writer
- `doc_id`: Document identifier
- `nsubj`: Subject of the sentence
- `nsubj_pos`: POS tag of the subject
- `root`: Root verb lemma
- `dative`: Dative object
- `dative_pos`: POS tag of dative object
- `direct_obj`: Direct object
- `direct_obj_pos`: POS tag of direct object
- `length_dative`: Log-transformed length of dative object
- `length_direct_obj`: Log-transformed length of direct object
- `construction_type`: Either 'double_object' or 'prepositional'

## Dependencies
- pandas: Data manipulation and analysis
- spacy: Natural language processing
- numpy: Numerical computations
- en_core_web_trf: spaCy transformer model for English

## Requirements
- GPU recommended for processing large corpora
- Sufficient RAM for loading and processing text data
- Python 3.8+

## Data Format
The input CSV files should contain:
- **metadata_with_text.csv**: Document metadata with columns including 'Native_language', 'docid_field'
- **text_only.csv**: Text data with 'text_field' column containing the actual text

## Author
Fatih Ünal Bozdağ - fbozdag1989@gmail.com
