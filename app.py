"""
Emotion Detection from Text  --  Streamlit web application.

The app only LOADS the model that `train_model.py` already saved, so a
prediction takes a few milliseconds instead of retraining every time.

Run it with:
    streamlit run app.py
"""

import json
import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from preprocessing import clean_text

# --------------------------------------------------------------------------
# File paths
# --------------------------------------------------------------------------
MODELS_DIR = "models"
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.pkl")
INFO_PATH = os.path.join(MODELS_DIR, "model_info.json")
CSS_PATH = os.path.join("assets", "style.css")

# --------------------------------------------------------------------------
# One emoji + one soft background tint for each of the 13 dataset labels.
# The labels below are the actual values in the `sentiment` column of the
# Kaggle dataset. Anything unexpected falls back to DEFAULT_STYLE.
# --------------------------------------------------------------------------
EMOTION_STYLE = {
    "happiness":  ("😄", "#FFF6DA"),
    "love":       ("🥰", "#FDE6EC"),
    "fun":        ("😜", "#FFF0DC"),
    "enthusiasm": ("🤩", "#FFF3D6"),
    "relief":     ("😌", "#E7F5EC"),
    "surprise":   ("😲", "#EFE8FB"),
    "neutral":    ("😐", "#F1F0EC"),
    "empty":      ("😶", "#EFEEEA"),
    "boredom":    ("🥱", "#EDEFF3"),
    "worry":      ("😟", "#E8F0FA"),
    "sadness":    ("😢", "#E4EDF8"),
    "hate":       ("😠", "#FBE7E2"),
    "anger":      ("😡", "#FCE1DC"),
}
DEFAULT_STYLE = ("💭", "#F5F3EE")


def style_for(label):
    """Return (emoji, tint colour) for an emotion label."""
    return EMOTION_STYLE.get(str(label).lower(), DEFAULT_STYLE)


# Ready-made sentences so the examiner can try the app in one click
EXAMPLES = [
    ("🎉 Good news", "I finally got my dream job and I cannot stop smiling!"),
    ("✈️ Excited", "I am so excited for my trip tomorrow!"),
    ("😟 Nervous", "I am really worried about my result tomorrow."),
    ("💔 Missing", "I miss my best friend so much today."),
]

# The five stages of the pipeline, drawn in the "How it works" section
STEPS = [
    ("✍️", "Enter text"),
    ("🧹", "Clean text"),
    ("🔢", "TF-IDF"),
    ("🧠", "Logistic Regression"),
    ("💭", "Emotion"),
]

# Little hand-drawn speech bubble for the header
HERO_DOODLE = """
<svg class="doodle" width="132" height="78" viewBox="0 0 132 78" fill="none"
     xmlns="http://www.w3.org/2000/svg">
  <path d="M24 11 C11 12, 7 21, 8 32 C9 44, 17 51, 31 51 L39 51 L35 66 L53 51 L93 50
           C107 49, 115 42, 114 30 C113 18, 105 11, 91 11 Z"
        stroke="#2E2A27" stroke-width="2.6" stroke-linejoin="round" fill="#FFFFFF"/>
  <circle cx="48" cy="28" r="3.1" fill="#2E2A27"/>
  <circle cx="74" cy="28" r="3.1" fill="#2E2A27"/>
  <path d="M45 36 C53 45, 69 45, 77 36" stroke="#EF7A5A" stroke-width="2.8" stroke-linecap="round"/>
  <path d="M122 13 l0 9 M117.5 17.5 l9 0" stroke="#7FB8AE" stroke-width="2.4" stroke-linecap="round"/>
  <path d="M13 59 l0 7 M9.5 62.5 l7 0" stroke="#EF7A5A" stroke-width="2.2" stroke-linecap="round"/>
  <path d="M105 63 l0 6 M102 66 l6 0" stroke="#7FB8AE" stroke-width="2" stroke-linecap="round"/>
</svg>
"""


# --------------------------------------------------------------------------
# Loading the stylesheet and the trained model
# --------------------------------------------------------------------------
def load_css():
    """Inject assets/style.css into the page."""
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load the saved vectorizer, model and evaluation scores.

    @st.cache_resource means this runs only once per session, not on every
    click, so predictions stay fast.
    """
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    info = {}
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH) as f:
            info = json.load(f)

    return vectorizer, model, info


def models_are_missing():
    return not (os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH))


# --------------------------------------------------------------------------
# The prediction itself
# --------------------------------------------------------------------------
def predict_emotion(text, vectorizer, model):
    """Clean -> TF-IDF -> Logistic Regression -> label + probabilities.

    Returns None if nothing usable is left after cleaning (for example if the
    user typed only numbers or symbols).
    """
    cleaned = clean_text(text)
    if not cleaned:
        return None

    features = vectorizer.transform([cleaned])       # text  -> TF-IDF vector
    probabilities = model.predict_proba(features)[0]  # one probability per class

    ranking = probabilities.argsort()[::-1]           # highest probability first
    classes = model.classes_

    return {
        "label": str(classes[ranking[0]]),
        "confidence": float(probabilities[ranking[0]]),
        "top3": [(str(classes[i]), float(probabilities[i])) for i in ranking[:3]],
    }


# --------------------------------------------------------------------------
# Small UI helpers (each returns / draws one piece of the page)
# --------------------------------------------------------------------------
def draw_hero():
    # NOTE: the HTML below is deliberately NOT indented. Markdown turns any
    # line indented by 4+ spaces into a code block, which would print the raw
    # <div> tags on screen instead of drawing them. Same for every other block
    # of raw HTML in this file.
    st.markdown(
        f"""
<div class="hero">
{HERO_DOODLE}
<h1>Emotion Detection from Text</h1>
<p>Type it. Express it. Let the model guess the feeling.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def draw_result(result):
    """The big result card: emoji, emotion name and confidence bar."""
    emoji, tint = style_for(result["label"])
    percent = result["confidence"] * 100

    st.markdown(
        f"""
<div class="result-card" style="background:{tint};">
<div class="cap">Detected Emotion</div>
<div class="emoji">{emoji}</div>
<div class="label">{result["label"]}</div>
<div class="conf-bar"><span style="width:{percent:.1f}%;"></span></div>
<div class="conf">Model confidence: {percent:.1f}%</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption(
        "Confidence is the probability the Logistic Regression model assigned to "
        "this emotion — it is the model's own estimate, not a guarantee."
    )

    # The two runner-up emotions, so you can see how close the call was
    runners = result["top3"][1:]
    if runners:
        rows = ""
        for label, prob in runners:
            r_emoji, _ = style_for(label)
            rows += (
                f'<div class="runner">{r_emoji} {label}'
                f'<span class="track"><span style="width:{prob * 100:.1f}%;"></span></span>'
                f"{prob * 100:.1f}%</div>"
            )
        st.markdown(
            f'<div class="doodle-box"><b>Other possibilities</b>{rows}</div>',
            unsafe_allow_html=True,
        )


def draw_history():
    """Prediction history for the current session (no database involved)."""
    history = st.session_state.history
    st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)

    if not history:
        st.markdown(
            '<p class="hint">Nothing here yet — your predictions will be collected below.</p>',
            unsafe_allow_html=True,
        )
        return

    # Newest first
    for item in reversed(history):
        shown = item["text"] if len(item["text"]) <= 110 else item["text"][:110] + "..."
        shown = shown.replace("<", "&lt;").replace(">", "&gt;")   # keep raw HTML out
        st.markdown(
            f"""
<div class="history-item">
<div class="quote">"{shown}"</div>
<div class="meta">{item["emoji"]} {item["label"].capitalize()}
&nbsp;—&nbsp; {item["confidence"] * 100:.1f}%
&nbsp;·&nbsp; {item["time"]}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.button("🧽 Clear History", on_click=clear_history)


def draw_how_it_works():
    steps_html = ""
    for number, (icon, text) in enumerate(STEPS, start=1):
        if number > 1:
            steps_html += '<div class="step-arrow">➜</div>'
        steps_html += (
            f'<div class="step"><div class="ico">{icon}</div>'
            f'<div class="txt">{text}</div><div class="num">step {number}</div></div>'
        )
    st.markdown(f'<div class="steps">{steps_html}</div>', unsafe_allow_html=True)

    st.markdown(
        """
**What actually happens**

1. **Enter text** — you type a sentence.
2. **Clean text** — it is lowercased; links, `@mentions`, digits and punctuation are removed.
3. **TF-IDF** — each word (and each two-word phrase) becomes a number showing how
   important it is in this sentence compared with the whole dataset.
4. **Logistic Regression** — the model weighs those numbers and produces a
   probability for each of the 13 emotions.
5. **Emotion** — the emotion with the highest probability is shown as the answer.
        """
    )


def draw_about_dataset():
    st.markdown(
        """
**Emotion Detection from Text** — by *Pashupati Gupta* on Kaggle
([dataset page](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text)).

- **40,000** tweets, **3** columns: `tweet_id`, `sentiment`, `content`
- Input feature: `content` (the tweet) · Target: `sentiment` (the emotion)
- **13** emotion classes: neutral, worry, happiness, sadness, love, surprise,
  fun, relief, hate, empty, enthusiasm, boredom, anger
- After removing duplicate tweets **39,827** records remain; a further 94 rows
  are dropped because they are only emoji or digits, leaving **39,733** rows
  that are split into 31,786 training and 7,947 test tweets.

The classes are very imbalanced — `neutral` has 8,638 tweets while `anger` has
only 110 — which is the main reason the rare emotions are hard to predict.
        """
    )


def draw_about_model(info):
    if not info:
        st.info("Run `python train_model.py` to generate the evaluation report.")
        return

    scores = info["scores"]
    st.markdown(
        f"""
**Pipeline:** TF-IDF (unigrams + bigrams) → Logistic Regression
&nbsp;·&nbsp; trained on {info["n_train"]:,} rows, tested on {info["n_test"]:,} unseen rows
&nbsp;·&nbsp; {info["n_features"]:,} TF-IDF features
&nbsp;·&nbsp; last trained {info["trained_at"]}

| Metric | Weighted average | Macro average |
|---|---|---|
| Accuracy | **{scores["accuracy"] * 100:.2f}%** | — |
| Precision | {scores["precision_weighted"] * 100:.2f}% | {scores["precision_macro"] * 100:.2f}% |
| Recall | {scores["recall_weighted"] * 100:.2f}% | {scores["recall_macro"] * 100:.2f}% |
| F1-score | {scores["f1_weighted"] * 100:.2f}% | {scores["f1_macro"] * 100:.2f}% |

*Weighted* averaging counts every tweet equally, so it is dominated by the common
emotions. *Macro* averaging counts every emotion equally, so it is much lower —
that gap is exactly the class-imbalance problem described above.
        """
    )

    with st.expander("Confusion matrix (rows = actual, columns = predicted)"):
        matrix = pd.DataFrame(
            info["confusion_matrix"], index=info["labels"], columns=info["labels"]
        )
        st.dataframe(matrix)


# --------------------------------------------------------------------------
# Session-state callbacks
#
# These run BEFORE the page is redrawn, which is the safe place to change the
# value of a widget that already has a key.
# --------------------------------------------------------------------------
def use_example(text):
    st.session_state.user_text = text


def clear_text():
    st.session_state.user_text = ""
    st.session_state.last_result = None


def clear_history():
    st.session_state.history = []


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Emotion Detection from Text", page_icon="🎭")
    load_css()
    draw_hero()

    # --- if the model has not been trained yet, stop here with a friendly note
    if models_are_missing():
        st.markdown(
            '<div class="doodle-box">'
            "<b>The model has not been trained yet.</b><br><br>"
            "Run this once in your terminal:<br>"
            "<code>python train_model.py</code><br><br>"
            "…or just press the button below (it takes about half a minute)."
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("🚀 Train the model now", type="primary"):
            from train_model import train_and_save  # imported only when needed
            try:
                with st.spinner("Training on ~40,000 tweets — please wait…"):
                    train_and_save()
            except (FileNotFoundError, ValueError) as error:
                st.error(str(error))
                st.stop()
            load_artifacts.clear()
            st.success("Done! The model is ready.")
            st.rerun()
        st.stop()

    try:
        vectorizer, model, info = load_artifacts()
    except Exception:
        st.error(
            "The saved model files could not be loaded. They may be from an "
            "older version of scikit-learn. Please delete the `models/` folder "
            "and run `python train_model.py` again."
        )
        st.stop()

    # --- session state: everything lives only for this browser session -------
    st.session_state.setdefault("user_text", "")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_result", None)

    # --- example sentences ---------------------------------------------------
    st.markdown('<p class="hint">Try one of these →</p>', unsafe_allow_html=True)
    for column, (caption, sentence) in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        column.button(
            caption,
            key=f"example_{caption}",
            on_click=use_example,
            args=(sentence,),
            use_container_width=True,
        )

    # --- the text box --------------------------------------------------------
    st.text_area(
        "What are you feeling today?",
        key="user_text",
        height=140,
        placeholder="Type something here...",
    )

    text = st.session_state.user_text
    words = len(text.split())
    st.markdown(
        f'<div class="counter">{len(text)} characters · {words} '
        f'{"word" if words == 1 else "words"}</div>',
        unsafe_allow_html=True,
    )

    # --- buttons -------------------------------------------------------------
    left, right = st.columns([2, 1])
    detect_clicked = left.button("✨ Detect Emotion", type="primary", use_container_width=True)
    right.button("🧼 Clear Text", on_click=clear_text, use_container_width=True)

    # --- run a prediction ----------------------------------------------------
    if detect_clicked:
        entered = st.session_state.user_text.strip()

        if not entered:
            st.warning("Please type something first — the box is empty!")
        else:
            with st.spinner("Reading between the lines…"):
                result = predict_emotion(entered, vectorizer, model)

            if result is None:
                st.warning(
                    "I could not find any words in that. Try a sentence with "
                    "actual words in it!"
                )
            else:
                st.session_state.last_result = result
                emoji, _ = style_for(result["label"])
                st.session_state.history.append(
                    {
                        "text": entered,
                        "label": result["label"],
                        "emoji": emoji,
                        "confidence": result["confidence"],
                        "time": datetime.now().strftime("%H:%M:%S"),
                    }
                )

    # --- show the most recent result (survives other button clicks) ----------
    if st.session_state.last_result:
        draw_result(st.session_state.last_result)

    # --- history -------------------------------------------------------------
    draw_history()

    # --- information sections ------------------------------------------------
    st.markdown('<div class="section-title">Good to know</div>', unsafe_allow_html=True)

    with st.expander("🛠️  How it works"):
        draw_how_it_works()

    with st.expander("📚  About the dataset"):
        draw_about_dataset()

    with st.expander("📈  About the model"):
        draw_about_model(info)

    st.markdown(
        '<p class="hint" style="text-align:center;margin-top:1.4rem;">'
        "Made with TF-IDF, Logistic Regression and a lot of doodles ✏️</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
