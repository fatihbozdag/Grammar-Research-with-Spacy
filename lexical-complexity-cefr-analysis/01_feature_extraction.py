import spacy
import lftk
import torch
torch.device("mps")
spacy.require_gpu()
nlp = spacy.load('en_core_web_trf', disable=['ner', 'textcat'])
from thinc.api import get_current_ops
get_current_ops()
import pandas as pd
df = pd.read_csv('sample_data.csv')  # EFCAMDAT corpus is available upon request here "https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html"

# 1. Create a function to process a single text and return the extracted features.
def process_text(text):
    doc = nlp(text)
    LFTK = lftk.Extractor(docs=doc)
    searched_features = lftk.search_features(domain="lexico-semantics", return_format="list_key")

    
    extracted_features = LFTK.extract(features=searched_features)
    
    # Convert the dictionary to a DataFrame and transpose it to have one row per text
    extracted_features_df = pd.DataFrame([extracted_features])
    
    return extracted_features_df
# 2. Use the `apply` method to apply this function to the column containing the texts in your DataFrame.
df['extracted_features'] = df['text_corrected'].astype(str).apply(process_text)

# 3. Combine the extracted features from each row into a new DataFrame.
extracted_features_df = pd.concat(df['extracted_features'].tolist()).reset_index(drop=True)

# Merge the original DataFrame with the extracted features DataFrame
df.reset_index(drop=True, inplace=True)
extracted_features_df.reset_index(drop=True, inplace=True)

df = pd.concat([df, extracted_features_df], axis=1)

# You can now use extracted_features_df or merged_df to analyze the extracted features.
print("Feature extraction completed!")
print(f"Total features extracted: {len(extracted_features_df.columns)}")
print(f"Dataset shape: {df.shape}")

# Save the results
df.to_csv('extracted_lexical_features.csv', index=False)
print("Results saved to 'extracted_lexical_features.csv'")
