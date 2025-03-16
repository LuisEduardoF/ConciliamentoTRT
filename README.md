<div align="center">
<table>
  <tr>
    <td align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Marca_Ufes_SVG.svg/1200px-Marca_Ufes_SVG.svg.png" alt="UFES" width="180"/></td>
    <td align="center"><img src="https://dsl.ufes.br/wp-content/uploads/2023/04/7-e1680834768938.png" alt="DSL" width="180"/></td>
  </tr>
</table>
</div>

# Predicting Conciliation Outcomes in court decisions at Vara de Trabalho of Espírito Santo

This repository presents a project for predicting whether a judicial decision (Julgamento) in the labor courts of Espírito Santo will end in a **Conciliação**. Inspired by prior work in legal decision prediction, this project adapts and extends existing methodologies to a new dataset and target outcome.

## Introduction

The project focuses on the prediction of case outcomes for labor court decisions in Espírito Santo. Unlike previous studies that dealt with multiple decision labels, our primary goal is to determine if a case will be resolved by conciliation. The dataset used contains detailed case information, ranging from procedural identifiers to case specifics, which are key to building robust predictive models.

## Objective

The main objective is to compare three different approaches to predict the outcome of a case as **Conciliação** or not. The project aims to:

- Provide a baseline using classical machine learning techniques.
- Leverage large language models to extract deep semantic features.
- Evaluate the performance of an end-to-end LLM classification pipeline.

The outcome of interest is whether the case concludes with **Conciliação**.

## Methodology

The dataset for this project, derived from *"Julgamentos da vara de trabalho no Espírito Santo"*, includes the following columns:

- **NÚMERO DO PROCESSO**
- **CLASSE PROCESSUAL**
- **VARA DO TRABALHO**
- **MAGISTRADO**
- **ASSUNTOS**
- **PORTADOR DEFICIÊNCIA**
- **SEGREDO DE JUSTIÇA**
- **RECDA ATIVA-INATIVA**
- **RECDA PES FÍS OU JUR**
- **OAB**
- **VALOR DA CAUSA**
- **RAMO DE ATIVIDADE**
- **CIDADE ORIG PET INICIAL**
- **ENTE PUB OU PRIV**
- **INDICADOR DO PROC**
- **QTD RTE**
- **QTD RDO**
- **TIPO DE SOLUÇÃO**
- **DATA DE JULGAMENTO**
- **DATA DE AJUIZAMENTO**
- **DOCUMENTOS DAS RECLAMADAS**
- **DOCUMENTOS DOS RECLAMANTES**

We propose a three-pronged approach:

1. **Baseline Approach (Classical Machine Learning):**  
   - **Models:** Random Forest, Logistic Regression, and Gradient Boosting.  
   - **Preprocessing:** Categorical variables are processed using Target Encoding.  
   - **Goal:** Establish a robust baseline for predicting if a decision ends as **Conciliação**.

2. **LLM Embedding Approach:**  
   - **Process:** Use a large language model (LLM) to generate embeddings from the textual components of the case data.  
   - **Application:** The embeddings are then fed into a traditional classifier.  
   - **Goal:** Leverage semantic features from the text to potentially improve prediction accuracy.

3. **LLM Classification Approach:**  
   - **Method:** Directly use a classification model based on a large language model, which processes the input data end-to-end.  
   - **Goal:** Compare the performance of an LLM-based classifier with that of the embedding-based method and the classical baseline.

## How to Use It

1. **Setup:**  
   - Clone the repository.
   - Install the required dependencies from `requirements.txt`.
   - Configure any necessary API keys if using external LLM services.

2. **Data Preparation:**  
   - Ensure the dataset is available in the correct format.
   - Preprocess the data:
     - Handle missing values.
     - Apply Target Encoding for categorical variables (for the baseline approach).
     - Split the dataset into training and testing sets.

3. **Model Training and Evaluation:**  
   - Run the provided scripts to train the baseline models (Random Forest, Logistic Regression, and Gradient Boosting).
   - For the LLM-based approaches:
     - Execute the embedding extraction pipeline, then train a classifier on these embeddings.
     - Alternatively, run the end-to-end LLM classification script.
   - Evaluate each model using metrics such as accuracy and F1-score.
   - Compare the performance across all three approaches.

4. **Deployment:**  
   - Use the provided notebooks or scripts to reproduce the experiments.
   - Adjust configuration parameters as needed for deployment.

## References

- Lage-Freitas, A., Allende-Cid, H., Santana, O., & de Oliveira-Lage, L. (2019). *Predicting Brazilian Court Decisions.* [arXiv:1905.10348](https://arxiv.org/abs/1905.10348)
- Additional literature on legal decision prediction and related machine learning techniques.
- Documentation for the libraries and tools used (e.g., scikit-learn, pandas, numpy, NLTK).
