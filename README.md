# Grammar Research with spaCy

## Overview
This repository contains a collection of NLP research projects focusing on grammar analysis in learner English using spaCy and other advanced NLP tools. Each project is organized as a separate module with its own documentation and requirements.

## Research Focus
I am a researcher working on NLP/Corpus studies, particularly exploring grammar patterns in learner English. This repository contains all the codes and data (where not copyrighted) corresponding to my published and ongoing studies.

## Projects

### 🔗 [Dative Alternation Analysis](./dative-alternation-analysis/)
**Probabilistic Analysis of English Dative Constructions in Academic Writings of English EFL Learners**
- Analyzes double object vs. prepositional dative constructions
- **Published:** *Theory and Practice of Second Language Acquisition*, 10(1), 1–24 (2024)
- **DOI:** https://doi.org/10.31261/TAPSLA.13902
- Uses spaCy for linguistic annotation and statistical analysis

### 🔗 [Modal Patterns Analysis](./modal-patterns-analysis/)
**Bayesian Analysis of Modal Verb Patterns in Learner English**
- Extracts five different modal verb construction patterns
- **Published:** *Heliyon*, 10(7):e28701 (2024)
- **DOI:** https://doi.org/10.1016/j.heliyon.2024.e28701
- Uses Bayesian statistical modeling with Bambi and PyMC
- Includes cross-linguistic comparison analysis

### 🔗 [Frame Semantic Tagging](./frame-semantic-tagging/)
**Automatic Frame Semantic Analysis for Target Lemmas**
- Uses Frame Semantic Transformer for automatic frame detection
- Extracts frame elements and semantic roles
- Processes bulk text data efficiently

### 🔗 [Topic Modeling and Lexical Representation](./topic-modeling-lexical-representation/)
**Cognitive Patterns in Learner Texts: Exploring Semantic and Semiotic Dimensions Through Topic Modeling**
- **Published:** IGI Global Scientific Publishing (2025)
- **DOI:** https://doi.org/10.4018/979-8-3693-8146-5.ch011
- Combines BERTopic for topic discovery with Word2Vec for lexical analysis
- Optional ChatGPT integration for topic representation
- Creates semantic networks and entropy analysis

### 🔗 [Lexical Complexity CEFR Analysis](./lexical-complexity-cefr-analysis/)
**Lexical Complexity and Language Proficiency: An Investigation of Indices Across CEFR Levels**
- **Published:** Cambridge Scholars Publishing (2024)
- **Co-authored with:** Abdurrahman Kılımcı
- Uses LFTK for lexical complexity feature extraction
- Applies LASSO and Random Forest for CEFR level prediction
- Analyzes cross-linguistic complexity patterns

## Getting Started

Each project is self-contained with:
- 📁 **main.py**: Core analysis script
- 📄 **README.md**: Detailed project documentation
- 📋 **requirements.txt**: Python dependencies
- 📊 **sample_data.csv**: Example data format

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/Grammar-Research-with-Spacy.git
cd Grammar-Research-with-Spacy

# Install dependencies for a specific project
cd dative-alternation-analysis
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## Research Applications
These tools are designed for:
- Corpus linguistics research
- Second language acquisition studies
- Grammar pattern analysis in learner corpora
- Cross-linguistic comparison studies
- Educational technology applications

## Dependencies
Each project has its own requirements, but common dependencies include:
- **spaCy**: Natural language processing
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Visualization

## Data Requirements
- ICLE (International Corpus of Learner English) format preferred
- CSV files with text and metadata columns
- GPU recommended for large corpus processing

## Citation
If you use any of these tools in your research, please cite the corresponding papers and acknowledge this repository.

## Contact
**Fatih Ünal Bozdağ**  
📧 fbozdag1989@gmail.com

Feel free to use, modify, and comment on these tools. Contributions and suggestions are welcome!
