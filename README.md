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

![Model Comparation](static/3.drawio.svg)

## Results

- **Traditional Models:** Achieved F1-macro scores between 0.60 and 0.65.
- **LLM Approaches:** 
  - The feature extraction method reached scores around 0.67.
  - The LLM classification approach achieved scores frequently above 0.70, demonstrating improved performance and lower dispersion in results.
 
![F1 model 1](static/f1_modelo.png)

![Model Comparation](static/f1_tempo.png)


## Conclusions

The study demonstrates that incorporating LLMs into judicial prediction models significantly enhances performance compared to classical approaches. Although LLMs introduce challenges such as increased computational costs and complexity, their ability to capture semantic nuances makes them a promising tool for aiding judicial decision-making.

## References

- Lage-Freitas, A., Allende-Cid, H., Santana, O., & de Oliveira-Lage, L. (2019). *Predicting Brazilian Court Decisions.* [arXiv:1905.10348](https://arxiv.org/abs/1905.10348)
- Additional literature on legal decision prediction and related machine learning techniques.
- Documentation for the libraries and tools used (e.g., scikit-learn, pandas, numpy, NLTK).

## Running the Project

### Locally on a Linux Terminal

To run the project on your Linux machine, follow these steps:

1. **Clone the repository:**

   ```bash
     git clone https://github.com/LuisEduardoF/Conciliamento_TRT.git
     cd Conciliamento_TRT
   ```
2. **Install dependecies:**
  Make sure you have Python (3.10) installed, then install the required packages:

  ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
   ```
3. **Run the analysis scripts:**
   ```bash
     python -u classical_analysis/model.py
     python -u llm_analysis/llm_embedding.py
     python -u llm_analysis/llm_classification.py
    ```
   This will execute the methodology and print the outputs directly in your terminal.

### Automated Execution with GitHub Actions

This repository leverages GitHub Actions to automatically run the analysis scripts whenever you push new commits or open a pull request on the `main` branch.

**How It Works:**

- **Trigger Conditions:**  
  The workflow is activated on any push or pull request targeting the `main` branch.

- **Python Environment:**  
  The job is executed using a matrix strategy that tests your code on Python 3.8, 3.9, and 3.10 to ensure compatibility.

- **Workflow Steps:**
  1. **Checkout Repository:**  
     The workflow starts by cloning your repository.
  2. **Setup Python:**  
     It sets up the designated Python version based on the matrix.
  3. **Install Dependencies:**  
     Pip is upgraded and the packages listed in `requirements.txt` are installed.
  4. **Execute Scripts:**  
     The following Python scripts are run in sequence using the unbuffered mode (`-u` flag) so that all print statements are immediately output:
     - `classical_analysis/model.py`
     - `llm_analysis/llm_embedding.py`
     - `llm_analysis/llm_classification.py`

- **Viewing Output:**  
  All the print outputs and logs are captured by GitHub Actions. To inspect them:
  1. Navigate to the **Actions** tab in your repository.
  2. Click on the desired workflow run.
  3. Expand each step to view the detailed stdout and stderr.
 [Action List](https://github.com/LuisEduardoF/Conciliamento_TRT/actions)
---

## Article Versioning

The project uses a versioning system for its article documentation, stored on the project’s wiki under **[VER] Articles versioning** ! [Article Versioning](https://github.com/LuisEduardoF/Conciliamento_TRT/wiki/%5BVER%5D-Articles-versioning). Each version corresponds to a specific stage in the development of the article:

- **(W) Work:**  
  Draft versions or college works articles.
- **(TH) Thesis:**  
  Versions that are formatted or refined for thesis submission.
- **(P) Publication:**  
  Finalized versions intended for publication.

For questions or further information, please contact Luís Eduardo Freire da Câmara at [luis.camara@edu.ufes.br].
