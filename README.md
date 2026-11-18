# BERTopic analysis of social media posts about alcohol and breast cancer

This repository contains the Python script used to preprocess social media posts, fit candidate BERTopic models, and evaluate computational topic coherence. The script fits 28 candidate models with requested topic counts from 3 to 30 and generates the coherence plot used for Figure 2. Researcher ratings, final theme development, and the manual sensitivity analysis of unassigned posts were conducted separately.

## Data availability

Infegy provided the study data under license. Requests for access may be directed to the corresponding author and are subject to Infegy’s permission. The repository contains the analysis code. The input workbook and outputs containing post text are available only as permitted under the data license.

## Running the analysis

Install the Python dependencies:

```bash
python -m pip install pandas openpyxl bertopic scikit-learn emoji gensim matplotlib
```

Place an authorized copy of the analysis input in the same folder as the script. Name the workbook `input_posts.xlsx` and ensure it contains a column named `Post`. From that folder, run:

```bash
python bertopic_analysis.py
```

BERTopic may download the embedding model on the first run. Topic assignments and coherence scores may vary across runs because the original script did not fix the random seed for dimensionality reduction.

## Outputs

The script writes one Excel workbook per candidate model, an overall coherence summary (`Overall_Coherence_Summary.xlsx`), and plots in the `coherence_plots/` folder. The per-model workbooks contain post text and topic assignments and should be handled under the same data-sharing restrictions as the input.
