# Arabic Functional Text Classification

Classifying Arabic text by its **function** (legal, instructional, academic, reviews, poetic, propaganda, and more) using Arabic-specific preprocessing, TF-IDF features and classical machine learning models, with a small **Gradio** demo.

Functional classification asks *what a text is for* (for example, to instruct, to persuade or to review), not just what it is about. This makes it harder than topic classification, because the same topic can appear in many functional styles.

> **Team project.** Built by a team of four students at BNU in 2025 as an NLP course project.

## Dataset

[`zeydferhat/arabic_functional_text_dimensions`](https://huggingface.co/datasets/zeydferhat/arabic_functional_text_dimensions) on Hugging Face, with **17 functional classes**:

`Instruct`, `Legal`, `Personal`, `academic`, `appell`, `argum`, `commercial`, `emotive`, `fictive`, `flippant`, `info`, `poetic`, `propaganda`, `report`, `reporting`, `reviews`, `specialist`

The original distribution is uneven, so a **balanced subset of 120 samples per class (2,040 texts)** is used. It is split 80/20 with stratification: 1,632 training and 408 test samples (24 per class).

![Label distribution](docs/label_distribution.png)

![Word cloud](docs/wordcloud.png)

## Pipeline

1. **Cleaning:** remove digits, punctuation and Arabic diacritics (tashkeel), normalize whitespace.
2. **Stopwords:** removal with a custom Arabic stopword list.
3. **Features:** TF-IDF on unigrams and bigrams.
4. **Models:** Multinomial Naive Bayes, Logistic Regression, SVM and Random Forest, each in a `Pipeline` with TF-IDF.
5. **Tuning:** `GridSearchCV` for Logistic Regression (C, L2 penalty) and Random Forest.
6. **Demo:** a Gradio interface for classifying text typed by the user.

ISRI stemming was implemented and tested, but it is **disabled in the final pipeline** because it lowered accuracy on this dataset.

The notebook also computes document-level statistical features (text length, average word length, punctuation count, digit ratio) scaled with `StandardScaler`. These are explored but **not used by the final models**, which use TF-IDF features only.

## Results

Test set of 408 samples, 17 balanced classes (random chance is about 6%):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Multinomial Naive Bayes | 0.64 | 0.65 |
| Logistic Regression | 0.72 | 0.71 |
| SVM (default RBF) | 0.70 | 0.70 |
| Random Forest (1000 trees) | 0.52 | 0.54 |
| **Logistic Regression, tuned (C=10, L2)** | **0.73** | **0.73** |

Per-class F1 for the tuned Logistic Regression:

- **Strongest:** Instruct (0.94), Legal (0.89), reviews (0.86), specialist (0.83), commercial (0.81)
- **Weakest:** fictive (0.53), reporting (0.61), argum (0.64), info (0.65)

Classes with overlapping style or purpose (for example `report` / `reporting` / `info`, or `emotive` / `appell`) are confused most often.

![Logistic Regression confusion matrix](docs/confusion_logistic_regression.png)

Confusion matrices for the other models are in [`docs/`](docs/).

## Demo

A Gradio app that classifies Arabic text typed by the user. Example: "هذا المطعم جميل" ("This restaurant is beautiful") is classified as `reviews`.

![Gradio demo](docs/gui.png)

The demo model in the notebook is a separate, simpler **TF-IDF + Logistic Regression** trained on the raw dataset without the cleaning or tuning steps above.

## How to Run

The notebook runs on **Google Colab** or locally.

```bash
pip install -r requirements.txt
jupyter notebook notebooks/arabic_functional_text_classification.ipynb
```

The dataset is downloaded automatically from Hugging Face through the `datasets` library. Run the cells in order. The last cell starts the Gradio interface.

## Repository Structure

```
.
├── notebooks/
│   └── arabic_functional_text_classification.ipynb
├── docs/                  # Figures used in this README
├── requirements.txt
└── README.md
```

## Tech Stack

Python · scikit-learn · NLTK · imbalanced-learn · Hugging Face Datasets · Pandas · Matplotlib · Seaborn · WordCloud · Gradio

## Limitations and Future Work

- Accuracy is moderate (73%). The dataset is small for 17 classes, and some classes are inherently close in style.
- Only Logistic Regression and Random Forest were tuned. SVM and Naive Bayes use their default settings.
- The statistical features were not combined with TF-IDF in the final models. Testing them properly (for example with `ColumnTransformer`) is a next step.
- Try Arabic transformer models such as AraBERT or CAMeLBERT, which usually do much better on Arabic text classification.
- Make the demo use the same preprocessing and tuned model as the evaluation pipeline.

## Acknowledgements

Dataset by `zeydferhat` on Hugging Face. Project supervised at BNU.
