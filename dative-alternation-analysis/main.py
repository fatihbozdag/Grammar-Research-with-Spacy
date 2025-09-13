### This is the code for my accepted paper titled 'Probabilistic Analysis of English Dative Constructions in Academic Writings of English EFL Learners' 
(January 2024 - Theory and Practice of Second Language Acquisition))###

import pandas as pd
import spacy
import math
import re

def load_data():
    # Load metadata and text files, preprocess and return as list of tuples
    # Add your own file paths here
    meta = pd.read_csv('F:/learner Corpora/metadata_with_text.csv')
    text = pd.read_csv('F:/Learner Corpora/text_only.csv')
    
    text['text_field'] = text['text_field'].apply(lambda x: re.sub(pattern, '', x).replace('\n', ''))
    meta_x = meta.to_dict('records')
    text_only = text['text_field'].values.tolist()
    icle = list(zip(text_only, meta_x))
    
    return icle
    
def setup_spacy():
    # Setup Spacy with required settings and return the instance
    spacy.require_gpu()
    pattern = r'ICLE\-\w+\-\w+\-\d+\.\d+'
    nlp = spacy.load('en_core_web_trf')
    nlp.max_length = 150000000000000
    nlp.create_pipe('merge_noun_chunks')
    nlp.add_pipe('merge_noun_chunks')
    
    return nlp

def is_valid_token(token, relative_pron_list, pos_list):
    return token.pos_ not in pos_list and token.lemma_ not in relative_pron_list

def process_double_object_dative(docs_with_context):
    # Process and extract information for double object dative constructions
    
    relative_pron_list = ['which', 'what', 'who', 'that', " "]
    pos_list = ['SPACE', 'X', 'SCON']
    
    dative_sentences = []
    native_language = []
    doc_id = []
    nsubj = []
    nsubj_pos = []
    root = []
    dative = []
    dative_pos = []
    direct_obj = []
    direct_obj_pos = []
    length_dative = []
    length_direct_obj = []

    for doc, context in docs_with_context:
        doc._.trf_data = None
        for d in doc:
            if d.dep_ == "dative" and d.pos_ != 'ADP' and d.head.pos_ == "VERB" and is_valid_token(d, relative_pron_list, pos_list):
                for n in d.head.children:
                    if n.dep_ == "nsubj" and is_valid_token(n, relative_pron_list, pos_list):
                        for x in d.head.children:
                            if x.dep_ == "dobj" and is_valid_token(x, relative_pron_list, pos_list):

                                dative.append(d.text)
                                dative_pos.append(d.pos_)
                                length_dative.append(math.log10(len(d.text)))
                                root.append(d.head.lemma_)
                                nsubj.append(n.text)
                                nsubj_pos.append(n.pos_)
                                direct_obj.append(x.text)
                                direct_obj_pos.append(x.pos_)
                                length_direct_obj.append(math.log10(len(x.text)))
                                dative_sentences.append(d.sent)
                                native_language.append(context['Native_language'])
                                doc_id.append(context['docid_field'])

    results = pd.DataFrame({
        'dative_sentences': dative_sentences,
        'native_language': native_language,
        'doc_id': doc_id,
        'nsubj': nsubj,
        'nsubj_pos': nsubj_pos,
        'root': root,
        'dative': dative,
        'dative_pos': dative_pos,
        'direct_obj': direct_obj,
        'direct_obj_pos': direct_obj_pos,
        'length_dative': length_dative,
        'length_direct_obj': length_direct_obj,
        'construction_type': 'double_object'
    })
    return results

def process_prepositional_dative(docs_with_context):
    # Process and extract information for prepositional dative constructions
    
    relative_pron_list = ['which', 'what', 'who', 'that', " "]
    pos_list = ['SPACE', 'X', 'SCON']
    
    dative_sentences_pre = []
    native_language_pre = []
    doc_id_pre = []
    nsubj_pre = []
    nsubj_pre_pos = []
    root_pre = []
    dative_pre = []
    dative_pre_pos = []
    direct_obj_pre = []
    direct_obj_pre_pos = []
    pre_obj = []
    pre_obj_pos = []
    length_pre_obj = []
    length_pre_direct_obj = []

    for doc, context in docs_with_context:
        doc._.trf_data = None
        for b in doc:
            if b.dep_ == "dative" and b.pos_ == "ADP" and b.head.pos_ == "VERB" and is_valid_token(b, relative_pron_list, pos_list):
                for m in b.head.children:
                    if m.dep_ == "nsubj" and is_valid_token(m, relative_pron_list, pos_list):
                        for k in b.head.children:
                            if k.dep_ == "dobj" and is_valid_token(k, relative_pron_list, pos_list):
                                for l in b.children:
                                    if l.dep_ == "pobj" and is_valid_token(l, relative_pron_list, pos_list):
                                        
                                        dative_sentences_pre.append(b.sent)
                                        native_language_pre.append(context['Native_language'])
                                        doc_id_pre.append(context['docid_field'])
                                        nsubj_pre.append(m.text)
                                        nsubj_pre_pos.append(m.pos_)
                                        root_pre.append(b.head.lemma_)
                                        dative_pre.append(b.text)
                                        dative_pre_pos.append(b.pos_)
                                        direct_obj_pre.append(k.text)
                                        direct_obj_pre_pos.append(k.pos_)
                                        length_pre_direct_obj.append(math.log10(len(k.text)))
                                        pre_obj.append(l.text)
                                        pre_obj_pos.append(l.pos_)
                                        length_pre_obj.append(math.log10(len(l.text)))

    results = pd.DataFrame({
        'dative_sentences': dative_sentences_pre,
        'native_language': native_language_pre,
        'doc_id': doc_id_pre,
        'nsubj': nsubj_pre,
        'nsubj_pos': nsubj_pre_pos,
        'root': root_pre,
        'dative': dative_pre,
        'dative_pos': dative_pre_pos,
        'direct_obj': direct_obj_pre,
        'direct_obj_pos': direct_obj_pre_pos,
        'pre_obj': pre_obj,
        'pre_obj_pos': pre_obj_pos,
        'length_dative': length_pre_obj,
        'length_direct_obj': length_pre_direct_obj,
        'construction_type': 'prepositional'
    })

    return results


def main():
    icle_data = load_data()
    nlp = setup_spacy()
    docs_with_context = list(nlp.pipe(icle_data, as_tuples=True))
    
    results_double_object_dative = process_double_object_dative(docs_with_context)
    results_prepositional_dative = process_prepositional_dative(docs_with_context)
    
    # Combine the DataFrames
    combined_results = pd.concat([results_double_object_dative, results_prepositional_dative], ignore_index=True)
    
    # Save, analyze or display the combined DataFrame as needed
    print(combined_results)

if __name__ == "__main__":
    main()

