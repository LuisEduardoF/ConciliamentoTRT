<div align="center">
<table>
  <tr>
    <td align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Marca_Ufes_SVG.svg/1200px-Marca_Ufes_SVG.svg.png" alt="UFES" width="180"/></td>
    <td align="center"><img src="https://dsl.ufes.br/wp-content/uploads/2023/04/7-e1680834768938.png" alt="DSL" width="180"/></td>
  </tr>
</table>
</div>

# Previsão de Resultados de Conciliação em Decisões Judiciais

This repository contains the code and analysis from the research project **"Previsão de Resultados de Conciliação em Decisões Judiciais na Vara do Trabalho do Espírito Santo"** by Luís Eduardo Freire da Câmara. The project investigates methods to predict conciliation outcomes in labor court cases using both classical machine learning techniques and state-of-the-art Large Language Models (LLMs).

## Overview

The Brazilian labor justice system faces a significant increase in case volume, with new processes rising by 28.7% in 2023. Conciliation plays a critical role in resolving cases efficiently; however, predicting which cases will be resolved via conciliation remains challenging. This work compares traditional natural language processing (NLP) techniques with LLM-based approaches to forecast whether a labor case will be settled through conciliation or by judicial sentence.

## Motivation

- **Judicial Overload:** With millions of cases, the system is under pressure.
- **Efficiency:** Early prediction of conciliation can help allocate judicial resources better.
- **Innovation:** Leveraging LLMs shows promise for capturing nuanced semantic information from legal texts, potentially outperforming traditional models.

## Data Description

The dataset consists of essential information for predicting conciliation outcomes in labor cases, organized into three main categories:

- **Identification and Classification:** Unique process identifiers and litigation types.
- **Economic and Party Characteristics:** Details such as the value of the claim, the nature of the parties (public or private), and the number of claimants/defendants.
- **Temporal and Outcome Data:** Dates of filing and judgment, and a binary indicator for conciliation (1) or judicial sentence (0).

Additional textual features include details like the Labor Court, Activity Branch, Procedural Class, Origin City, OAB information, subjects, and associated documents.

## Methodology

The project explores two primary approaches for predicting case outcomes:

### 1. Classical Machine Learning Approach (Baseline)
- **Preprocessing:** 
  - Removal of null entries and duplicates.
  - Conversion to lowercase, tokenization, and stopwords removal.
- **Windowing Strategy:** 
  - Data is divided into overlapping 2-year windows (with a 6-month shift) to capture temporal patterns.
- **Model Training:** 
  - Techniques such as Logistic Regression, Random Forest, and Gradient Boosting are applied.
  - Target encoding with smoothing is used to handle categorical text features.
  - Models are evaluated using metrics like accuracy, precision, recall, and F1-macro score.

### 2. Large Language Models (LLM) Approaches
- **LLM Feature Extraction:**
  - Texts are tokenized and processed using BERTimbau (a BERT model pre-trained on Portuguese) to extract high-dimensional embeddings.
  - A Random Forest classifier is then trained on these embeddings.
- **LLM Classification with HuggingFace Trainer:**
  - A pre-trained sequence classification model is fine-tuned using the HuggingFace Trainer.
  - This approach integrates tokenization, dataset conversion, and model training in one streamlined process.
![Comparação entre os modelos](static/Conciliamento.drawio.png)
## Results

- **Traditional Models:** Achieved F1-macro scores between 0.60 and 0.65.
- **LLM Approaches:** 
  - The feature extraction method reached scores around 0.67.
  - The LLM classification approach achieved scores frequently above 0.70, demonstrating improved performance and lower dispersion in results.
## Conclusions

The study demonstrates that incorporating LLMs into judicial prediction models significantly enhances performance compared to classical approaches. Although LLMs introduce challenges such as increased computational costs and complexity, their ability to capture semantic nuances makes them a promising tool for aiding judicial decision-making.

## References

- Lage-Freitas, A., Allende-Cid, H., Santana, O., & de Oliveira-Lage, L. (2019). *Predicting Brazilian Court Decisions.* [arXiv:1905.10348](https://arxiv.org/abs/1905.10348)
- Additional literature on legal decision prediction and related machine learning techniques.
- Documentation for the libraries and tools used (e.g., scikit-learn, pandas, numpy, NLTK).
