# Modal Patterns Analysis

## Overview
This project analyzes modal verb patterns in learner English corpora using spaCy for linguistic annotation and Bayesian statistical modeling with Bambi for probabilistic analysis.

## Research Context
This code corresponds to the published paper:

**Bozdağ FÜ, Morris G, Mo J. A Bayesian probabilistic analysis of the use of English modal verbs in L2 writing: Focusing on L1 influence and topic effects. *Heliyon*. 2024 Mar 27;10(7):e28701. doi: 10.1016/j.heliyon.2024.e28701. PMID: 38596125; PMCID: PMC11002062.**

📖 **Full article available at:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11002062/

## Features
- **Five Modal Pattern Types**: 
  - Pattern 1: Modal + Base Verb (e.g., "can go")
  - Pattern 2: Modal + Gerund (e.g., "can going") 
  - Pattern 3: Modal + Past Participle (e.g., "can gone")
  - Pattern 4: Modal + Passive with "be" (e.g., "can be done")
  - Pattern 5: Modal + Passive with "been" (e.g., "can have been done")
- **Bayesian Statistical Modeling**: Uses Bambi and PyMC for hierarchical Bayesian analysis
- **Cross-linguistic Analysis**: Compares modal patterns across different native languages
- **Visualization**: Creates density plots showing modal usage patterns

## Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## Usage
1. Prepare your data files:
   - `ICLE.csv`: Metadata file
   - `ICLE_text_only.csv`: Text data file
   - `all_data.csv`: Combined analysis data
   - `abortion_patterns.csv`: Specific topic analysis data

2. Update file paths in the `load_data()` function:
```python
meta = pd.read_csv('path/to/your/ICLE.csv')
text = pd.read_csv('path/to/your/ICLE_text_only.csv')
```

3. Run the modal pattern extraction:
```bash
python main.py
```

## Output
The script generates:
- **DataFrame with modal patterns**: Contains subject, modal, verb, and sentence information
- **Bayesian model results**: Statistical analysis of modal usage patterns
- **Visualization plots**: Density plots comparing modal patterns across languages

## Dependencies
- **NLP Processing**: pandas, spacy, numpy
- **Bayesian Modeling**: arviz, bambi, pymc, numpyro
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: scipy, sklearn

## Statistical Models
The analysis includes two main Bayesian models:
1. **Complete Dataset Model**: Analyzes all modal patterns across all topics
2. **Abortion Topic Model**: Focused analysis on abortion-related modal patterns

Both models use:
- Hierarchical structure with random effects for native language
- Categorical family for modal verb predictions
- Normal priors for all predictors

## Requirements
- GPU recommended for PyMC sampling
- Sufficient RAM for large corpus processing
- Python 3.8+

## Data Format
Input files should contain:
- **ICLE.csv**: Document metadata
- **ICLE_text_only.csv**: Text data with 'text_field' column
- **all_data.csv**: Analysis-ready data with modal patterns
- **abortion_patterns.csv**: Topic-specific modal patterns

## Visualization
The script generates comparative density plots showing:
- Modal usage patterns across Chinese and Turkish subcorpora
- Distribution of native language effects on modal semantic classes
- Posterior distributions for model parameters

## Author
Fatih Ünal Bozdağ - fbozdag1989@gmail.com
