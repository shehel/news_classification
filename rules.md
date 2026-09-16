# Introduction
The objective of this challenge is to classify Arabic news articles across six independent dimensions at the same time: relevance of the news article to Qatar, plus five additional categories.

# Evaluation
Submissions are automatically scored using F1-Score, computed independently for each of the six labels, and then averaged using a weighted average.

# Data
The dataset for this challenge consists of Arabic news articles sourced from major news platforms. The goal is to build a multi-task NLP model that can analyse Arabic text and predict six target dimensions simultaneously for each article:

    Qatar Related: A binary target indicating whether the article is genuinely centered on Qatar (1) or is an incidental mention / non-Qatari event (0).
    Geography: The geographic region associated with the article (G-1 through G-12, or NO_GEOGRAPHY if not applicable).
    Politics & Conflict: The political dimension of the content (PC-1,PC-2, or NO_POLITICS).
    Health & Wellbeing: The health-related dimension of the content (H-1,H-2, or NO_HEALTH).
    Science: The scientific and technological dimension of the content (SC-1,SC-2,SC-3, or NO_SCIENCE).
    Sports: The sporting dimension of the content (SP-1 through SP-4, or NO_SPORTS).

All categorical labels have been anonymized into opaque identifiers (G-*, PC-*, H-*, SC-*, SP-*), while the explicit absence of a specific domain is represented by NO_*. The Qatar Related target is provided as a binary label (0 or 1).

Your model must learn the semantic patterns and representations directly from the Arabic text and training annotations.

Your model must learn the semantic patterns and representations directly from the Arabic text and training annotations.

# Files
## Train.csv
Contains 3,786 labeled Arabic news articles with ID, text, and all six ground-truth target columns for model training and validation.

## arabert_starter_notebook_Qatar.ipynb
arabert starter notebook

## ml_starter_notebook_Qatar.ipynb
ml starter notebook

## Test.csv
Contains 1,262 unlabeled Arabic news articles (ID and text only). Apply your model to this dataset.

## SampleSubmission.csv
A template showing the exact format required for submission.

