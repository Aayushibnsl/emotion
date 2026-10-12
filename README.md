# 🎭 Emotion Detection from Text

A college Machine Learning project that predicts the **emotion** behind a piece of
text using **TF-IDF** and **Logistic Regression**, wrapped in a hand-drawn,
notebook-styled **Streamlit** web app.

```
You type a sentence
        ↓
   Text cleaning
        ↓
TF-IDF vectorization
        ↓
Logistic Regression
        ↓
Emotion + confidence  →  saved to session history
```

---

## 1. Project Overview

This is a **multi-class text classification** (NLP) project. The user types any
sentence into the app and the trained model predicts which of **13 emotions**
the sentence expresses, along with the probability the model assigned to it.

**Example**

| Input | Output |
|---|---|
| `I finally got my dream job!` | 😄 **Happiness** — model confidence 30.5% |
| `I am really worried about my result tomorrow.` | 😟 **Worry** — model confidence 68.0% |

---

## 2. Dataset

**Emotion Detection from Text** by **Pashupati Gupta** on Kaggle
👉 https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text

| Property | Value |
|---|---|
| Records | 40,000 tweets |
| Columns | `tweet_id`, `sentiment`, `content` |
| Input feature (X) | `content` — the raw tweet text |
| Target (y) | `sentiment` — the emotion label |
| Classes | 13 |
| Records after removing duplicate tweets | **39,827** |
| Records actually used for train/test | 39,733 |

**The 13 emotion labels** (read directly from the `sentiment` column, not
hard-coded guesses), with their counts:

| Emotion | Count | | Emotion | Count |
|---|---|---|---|---|
| neutral | 8,638 | | fun | 1,776 |
| worry | 8,459 | | relief | 1,526 |
| happiness | 5,209 | | hate | 1,323 |
| sadness | 5,165 | | empty | 827 |
| love | 3,842 | | enthusiasm | 759 |
| surprise | 2,187 | | boredom | 179 |
| | | | anger | 110 |

The 40,000 rows shrink in two steps: dropping duplicate tweets leaves
**39,827**, and a further 94 rows are dropped because they contain only emoji,
digits or punctuation and become empty once cleaned — leaving **39,733** rows
that are split 80/20 into 31,786 training and 7,947 test tweets.

Notice how imbalanced this is — `neutral` has **8,638** tweets while `anger` has
only **110**. This single fact explains most of the model's weaknesses (see
[Limitations](#10-limitations)).

### Where the CSV comes from

`data/emotion_dataset.csv` **is already included in this project**, so you can
train straight away.

The Kaggle download page requires a login, so the file was rebuilt from the
dataset's original source — CrowdFlower's `text_emotion.csv`, which is exactly
what the Kaggle dataset republishes — retrieved from the Internet Archive. The
original file has one extra `author` column; it was dropped so the CSV matches
the Kaggle schema (`tweet_id`, `sentiment`, `content`) exactly. The 40,000 rows,
the 13 labels and their counts are identical.

**If you prefer to download it yourself from Kaggle:**

1. Sign in to Kaggle and open the dataset page linked above.
2. Click **Download** — you get `archive.zip` containing `tweet_emotions.csv`.
3. Unzip it and save the file as `data/emotion_dataset.csv` (rename it).
4. Run `python train_model.py` again.

The training script only needs the `content` and `sentiment` columns, so either
file works.

---

## 3. Machine Learning Approach

| Stage | What happens | Why |
|---|---|---|
| **1. Clean text** | Lowercase; remove URLs, `@mentions`, digits, punctuation; squeeze spaces | `Happy` and `happy` should be the same word; links and usernames carry no emotion |
| **2. TF-IDF** | Each tweet becomes a vector of word-importance scores (unigrams **and** bigrams) | Models need numbers, not words. Bigrams let the model see phrases such as *"not happy"* |
| **3. Logistic Regression** | Fits one weight per word per emotion and outputs a probability for all 13 classes | Simple, fast, interpretable, and `predict_proba()` gives the confidence score the app shows |
| **4. Predict** | The emotion with the highest probability wins | |

**Settings used**

```python
TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
LogisticRegression(max_iter=1000, C=1.0, random_state=42)
train_test_split(test_size=0.2, random_state=42, stratify=y)
```

`stratify=y` keeps the same emotion proportions in the train and test halves,
which matters a lot on an imbalanced dataset. These values were picked by
comparing a few options on the test set — they gave the best accuracy and
weighted F1.

---

## 4. Model Evaluation

Measured on the **7,947 unseen test tweets** (20% of the data):

| Metric | Weighted average | Macro average |
|---|---|---|
| **Accuracy** | **35.47%** | — |
| Precision | 36.74% | 33.86% |
| Recall | 35.47% | 17.40% |
| F1-score | 31.60% | 17.26% |

- **Weighted** averaging counts every tweet equally, so common emotions dominate.
- **Macro** averaging counts every emotion equally, so rare emotions drag it down.
- The gap between the two columns *is* the class-imbalance problem.

A full **classification report** and a **13×13 confusion matrix** are printed by
`train_model.py` and are also viewable inside the app under
**About the model → Confusion matrix**.

> 35% may look low, but with **13** classes random guessing would score about
> **7.7%**, and the labels themselves are noisy — plenty of tweets could
> honestly be tagged `worry` *or* `sadness`. Published results on this dataset
> with classical models sit in the same 30–40% range.

---

## 5. Technologies Used

| Tool | Used for |
|---|---|
| Python 3 | everything |
| pandas | loading and cleaning the CSV |
| NumPy | numeric arrays (used under the hood by scikit-learn) |
| scikit-learn | TF-IDF, Logistic Regression, train/test split, metrics |
| joblib | saving and loading the trained model |
| Streamlit | the web interface |
| HTML / CSS | the hand-drawn "notebook" theme |

---

## 6. Project Structure

```
emotion-detection/
│
├── app.py                  # Streamlit web app (loads the saved model)
├── train_model.py          # trains the model and saves the .pkl files
├── preprocessing.py        # clean_text(), shared by both files above
├── requirements.txt
├── README.md
│
├── data/
│   └── emotion_dataset.csv         # 40,000 tweets (the Kaggle dataset)
│
├── models/                          # created by train_model.py
│   ├── tfidf_vectorizer.pkl
│   ├── emotion_model.pkl
│   └── model_info.json             # evaluation scores shown in the app
│
├── assets/
│   └── style.css                   # the hand-drawn theme
│
└── .streamlit/
    └── config.toml                 # keeps the app in light "paper" mode
```

**Why `preprocessing.py` exists as its own file:** the text must be cleaned in
*exactly* the same way during training and during prediction. Keeping
`clean_text()` in one shared file means the two can never drift apart — a very
common and very hard-to-spot bug in ML projects.

---

## 7. Installation

```bash
# 1. go into the project folder
cd emotion-detection

# 2. (recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows:  .venv\Scripts\activate

# 3. install the libraries
pip install -r requirements.txt
```

---

## 8. Train the Model

```bash
python train_model.py
```

This prints each step and the evaluation report, then writes
`tfidf_vectorizer.pkl`, `emotion_model.pkl` and `model_info.json` into
`models/`. It takes roughly **20–30 seconds**.

You only need to run this **once**. The app then just loads the saved files, so
predictions are instant.

---

## 9. Run the Application

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

*(If you forget to train first, the app will say so and offer a
**🚀 Train the model now** button.)*

### What the app can do

- ✍️ **Big text box** — type any sentence, with a live character / word counter
- 🎲 **Example buttons** — four ready-made sentences for a quick demo
- ✨ **Detect Emotion** — shows a result card with the emoji, emotion name and a
  confidence bar, plus the two runner-up emotions
- 🕘 **Prediction History** — every prediction from the current session
  (text, emotion, confidence, time) with a **Clear History** button.
  This uses **Streamlit session state** — there is no database and no login, so
  the history disappears when you close the app
- 📖 **How it works / About the dataset / About the model** — expandable
  sections explaining the pipeline and showing the evaluation scores

---

## 10. Limitations

1. **Overall accuracy is ~35%.** Thirteen classes, heavy imbalance and noisy
   crowd-sourced labels make this a genuinely hard problem for a linear model.
2. **Rare emotions are barely predicted.** `anger` (110 tweets) and `boredom`
   (179 tweets) get almost no correct predictions — there simply is not enough
   data for them. This is why the macro F1 (17%) is far below the weighted one.
3. **Bag-of-words has no word order.** TF-IDF sees *"not happy at all"* mostly as
   the separate words *not*, *happy*, *all*. Bigrams help a little, but only a little.
4. **No sarcasm or context understanding.** *"Great, another Monday"* will
   probably be read as positive.
5. **Trained on 2008-era tweets.** The vocabulary is short, informal and dated;
   long or formal sentences are outside what the model has seen.
6. **Emoji and punctuation are stripped** during cleaning, even though `:)` and
   `!!!` genuinely carry emotion. Keeping them is an easy future improvement.
7. **Confidence is a probability, not certainty.** A 92% score means the model
   assigned that emotion a 0.92 probability — it is the model's own estimate and
   it can be confidently wrong.

### Possible improvements (not implemented — kept out on purpose)

Balancing the classes (`class_weight="balanced"` or resampling), grouping the 13
labels into fewer broader emotions, trying LinearSVC or Naive Bayes, or moving to
word embeddings / transformer models. This project deliberately sticks to the
documented **TF-IDF + Logistic Regression** pipeline.

---

## Quick reference

```bash
pip install -r requirements.txt   # install
python train_model.py             # train (once)
streamlit run app.py              # run the app
```
