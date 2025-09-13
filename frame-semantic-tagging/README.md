# Frame Semantic Tagging

## Overview
This project performs automatic frame semantic analysis on text data using the Frame Semantic Transformer model. It identifies semantic frames and their associated frame elements for target lemmas in sentences.

## Features
- **Automatic Frame Detection**: Uses state-of-the-art frame semantic transformer model
- **Frame Element Extraction**: Identifies and extracts frame elements for detected frames
- **Target Lemma Analysis**: Focuses frame detection on specific target lemmas
- **Bulk Processing**: Efficiently processes large datasets
- **JSON Output**: Serializes frame elements as structured JSON data

## Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## Usage
1. Prepare your data file with the following columns:
   - `sent`: Sentence containing the target lemma
   - `lemma`: Target lemma to analyze for frame semantics

2. Update the data file path in the script:
```python
df = pd.read_csv('path/to/your/data_file.csv')
```

3. Run the frame semantic analysis:
```bash
python main.py
```

## Output
The script adds two new columns to your DataFrame:
- **`frames`**: Comma-separated list of detected frame names
- **`frame_elements`**: JSON string containing frame element mappings

### Example Output
```
frames: "Giving, Communication"
frame_elements: '{"Recipient": "the student", "Theme": "the book", "Donor": "the teacher"}'
```

## Dependencies
- **torch**: PyTorch for deep learning models
- **pandas**: Data manipulation and analysis
- **spacy**: Natural language processing for tokenization
- **frame-semantic-transformer**: Pre-trained frame semantic model

## Requirements
- GPU recommended for faster processing
- Python 3.8+
- Sufficient RAM for model loading and batch processing

## Data Format
Your input CSV should contain:
- **sent**: The sentence text (string)
- **lemma**: The target lemma to analyze (string)

Additional columns are preserved in the output.

## How It Works
1. **Sentence Processing**: Uses spaCy to tokenize and analyze sentences
2. **Lemma Matching**: Finds the character index of the target lemma in the sentence
3. **Frame Detection**: Applies the frame semantic transformer to detect frames
4. **Frame Filtering**: Only includes frames triggered by the target lemma
5. **Element Extraction**: Extracts frame elements for matching frames
6. **JSON Serialization**: Converts frame elements to JSON format

## Frame Semantic Analysis
Frame semantics is a linguistic theory that analyzes meaning in terms of structured conceptual frames. This tool helps researchers:
- Identify semantic roles in text
- Analyze verb argument structure
- Study semantic patterns in learner language
- Extract structured semantic information from unstructured text

## Performance Notes
- Processing time depends on sentence length and complexity
- GPU acceleration significantly improves performance
- Batch processing is more efficient than individual sentence analysis

## Author
Fatih Ünal Bozdağ - fbozdag1989@gmail.com
