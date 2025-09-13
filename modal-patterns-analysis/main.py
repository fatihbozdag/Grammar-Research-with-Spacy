
import spacy
import pandas as pd
import re
from typing import List, Tuple

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv('ICLE.csv')
    text = pd.read_csv('ICLE_text_only.csv')

    pattern = r'ICLE\-\w+\-\w+\-\d+\.\d+'
    text['text_field'] = text['text_field'].apply(lambda x: re.sub(pattern, '', x).replace('\n', ''))

    return meta, text

def process_texts(meta: pd.DataFrame, text: pd.DataFrame) -> List[Tuple[str, dict]]:
    meta_x = meta.to_dict('records')
    text_only = text['text_field'].values.tolist()
    return list(zip(text_only, meta_x))

def process_modal_pattern(meta_text: List[Tuple[str, dict]], pattern_function) -> pd.DataFrame:
    modal_pattern = []
    nlp = spacy.load('en_core_web_trf')

    for doc, context in nlp.pipe(meta_text, as_tuples=True):
        modal_pattern.extend(pattern_function(doc, context))

    return pd.DataFrame(modal_pattern)

def pattern1_function(doc, context):
    results = []
    for a in doc:
        if a.dep_ == 'aux' and a.tag_ == 'MD' and a.head.tag_ == 'VB':
            for b in a.head.children:
                if b.dep_ == 'nsubj':
                    result = {'Docid_field': context['docid_field'],
                              'Subject': b.text,
                              'Subject_Pos': b.pos_,
                              'Modal': a.text,
                              'Verb': a.head.text,
                              'Sent': a.sent}
                    results.append(result)
    return results
    
def pattern2_function(doc, context):
    results = []
    for token in doc:
        if token.dep_ == "aux" and token.tag_ == "MD" and token.head.tag_ == "VBG":
            for child in token.head.children:
                if child.dep_ == "nsubj":
                    result = {'Docid_field': context['docid_field'],
                              'Subject': child.text,
                              'Subject_Pos': child.pos_,
                              'Modal': token.text,
                              'Verb': token.head.text,
                              'Sent': token.sent}
                    results.append(result)
    return results


def pattern3_function(doc, context):
    results = []
    for token in doc:
        if token.dep_ == "aux" and token.tag_ == "MD" and token.head.tag_ == "VBN":
            for child in token.head.children:
                if child.dep_ == "nsubj":
                    result = {'Docid_field': context['docid_field'],
                              'Subject': child.text,
                              'Subject_Pos': child.pos_,
                              'Modal': token.text,
                              'Verb': token.head.text,
                              'Sent': token.sent}
                    results.append(result)
    return results


def pattern4_function(doc, context):
    results = []
    for token in doc:
        if token.dep_ == "ROOT" and token.tag_ == "VBN":
            for child in token.children:
                if child.dep_ == "aux" and child.tag_ == "MD":
                    for c in token.children:
                        if c.dep_ == "auxpass" and c.tag_ == "VB":
                            for d in token.children:
                                if d.dep_ == "nsubjpass":
                                    result = {'Docid_field': context['docid_field'],
                                              'Subject': d.text,
                                              'Subject_Pos': d.pos_,
                                              'Modal': child.text,
                                              'Verb': token.text,
                                              'Sent': token.sent}
                                    results.append(result)
    return results


def pattern5_function(doc, context):
    results = []
    for token in doc:
        if token.dep_ == "ROOT" and token.tag_ == "VBN":
            for child in token.children:
                if child.dep_ == "aux" and child.tag_ == "MD":
                    for c in token.children:
                        if c.dep_ == "auxpass" and c.tag_ == "VBN":
                            for d in token.children:
                                if d.dep_ == "nsubjpass":
                                    result = {'Docid_field': context['docid_field'],
                                              'Subject': d.text,
                                              'Subject_Pos': d.pos_,
                                              'Modal': child.text,
                                              'Verb': token.text,
                                              'Sent': token.sent}
                                    results.append(result)
    return results



def main():
    meta, text = load_data()
    icle = process_texts(meta, text)
    
    pattern1_df = process_modal_pattern(icle, pattern1_function)
    pattern2_df = process_modal_pattern(icle, pattern2_function)
    pattern3_df = process_modal_pattern(icle, pattern3_function)
    pattern4_df = process_modal_pattern(icle, pattern4_function)
    pattern5_df = process_modal_pattern(icle, pattern5_function)

    combined_df = pd.concat([pattern1_df, pattern2_df, pattern3_df, pattern4_df, pattern5_df], ignore_index=True)
    print(combined_df)

if __name__ == "__main__":
    main()

### Regression analysis code starts here ###

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt

plt.show(block=True)
plt.interactive(False)
import numpy as np
import pandas as pd
import seaborn as sns
import pymc as pm
import pymc.sampling_jax
import numpyro
import pickle

from matplotlib.lines import Line2D

print(f"Running on PyMC v{pm._version_}")

all_data = pd.read_csv('all_data.csv', encoding='utf-8') ### make sure you import your own csv file###

common_prior = bmb.Prior("Normal", mu = 0, sigma = 1)
priors = {"Subject_Pos": common_prior, "Pattern_Type": common_prior, "Modal_C": common_prior, 
          "Verb_C": common_prior, "Native_language": common_prior}

main_model = bmb.Model("Modal_C ~ 0 + Subject_Pos + Pattern_Type + Verb_C + (1|Native_language)",
                  data = all_data, family = "categorical",
                  auto_scale = True, priors = priors, categorical = ["Modal_C", "Verb_C", "Pattern_Type"])
main_model.build()

with main_model.backend.model:
    main_model_idata=pm.sampling_jax.sample_numpyro_nuts(draws= 2500, tune = 100, target_accept = .99, postprocessing_backend = 'gpu')
    posterior_predictive = pm.sample_posterior_predictive(trace = main_model_idata, extend_inferencedata=True)

abortion_data = pd.read_csv('abortion_patterns.csv')

common_prior = bmb.Prior("Normal", mu=0, sigma = 1)
priors = {"Subject_Pos": common_prior, "Native_language": common_prior, "Modal_C":common_prior,"Verb_C":common_prior, "Pattern_Type": common_prior}

abortion_model = bmb.Model("Modal_C~ 0 + Subject_Pos + Verb_C + Pattern_Type + (1|Native_language)",
                           data=abortion_data, categorical=["Modal_C", "Verb_C", "Pattern_Type"], family="categorical", auto_scale=True, priors = priors)

abortion_model.build()

with abortion_model.backend.model:
    abortion_idata=pm.sampling_jax.sample_numpyro_nuts(draws= 2500, tune = 100, target_accept = .99, postprocessing_backend = 'gpu')
    posterior_predictive = pm.sample_posterior_predictive(trace = abortion_idata, extend_inferencedata=True)


loo_abortion = az.loo(abortion_idata, pointwise = True).pareto_k
loo_patterns = az.loo(main_model_idata, pointwise = True).pareto_k


axes = az.plot_density(
    [main_model_idata, abortion_idata], 
    data_labels=["Complete Dataset", "Abortion Dataset"],
    var_names=["1|Native_language"], 
    hdi_prob=0.89,
    colors=["cornflowerblue", "sandybrown"],
    point_estimate="median", 
    shade=0.6, 
    grid=(2, 2)
)

# Add titles to each subplot for clarity
axes[0, 0].set_title('Panel (a): Chinese Subcorpus, per/pos/abi modals')
axes[0, 1].set_title('Panel (b): Chinese Subcorpus, vol/pre modals')
axes[1, 0].set_title('Panel (c): Turkish Subcorpus, per/pos/abi modals')
axes[1, 1].set_title('Panel (d): Turkish Subcorpus, vol/pre modals')

fig = axes.flatten()[0].get_figure()
fig.align_labels()

# Refined labels for the axes based on the context of KDE plots and your data
x_label = "Point Estimate (Median)"
y_label = "Density"

# Assigning the refined labels to each subplot in the 2x2 grid
for ax in axes.flatten():
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

# Consider revising the figure supertitle and caption to more accurately describe the plots and data being represented
fig.text(0.5, 0.02, "Distribution of estimated influence of native language on modal semantic classes in Chinese and Turkish subcorpora, visualized through kernel density estimation.", ha="center", va="center", wrap=True)

# The figure supertitle and caption provided above are suggestions; you should adjust them to fit the exact details and findings of your study.


