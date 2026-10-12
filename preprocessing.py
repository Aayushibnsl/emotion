"""
Text preprocessing for the Emotion Detection from Text project.

Both `train_model.py` and `app.py` import `clean_text()` from this file, so the
text is cleaned in EXACTLY the same way during training and during prediction.
If the two ever cleaned text differently, the model would silently make worse
predictions in the app than it did during testing.
"""

import re

# Pre-compiled regular expressions (compiled once = faster on 40,000 rows)
URL_PATTERN = re.compile(r"http\S+|www\.\S+")     # links:      http://t.co/abc
MENTION_PATTERN = re.compile(r"@\w+")             # mentions:   @tiffanylue
NON_LETTER_PATTERN = re.compile(r"[^a-z\s']")     # digits, punctuation, emoji
EXTRA_SPACE_PATTERN = re.compile(r"\s+")          # runs of whitespace


def clean_text(text):
    """Turn a raw tweet into a simple lowercase string of words.

    Steps:
        1. Lowercase everything          -> "Happy" and "happy" become one word
        2. Remove URLs                   -> links carry no emotion
        3. Remove @mentions              -> usernames are not emotion words
        4. Keep only letters and '       -> drops digits, punctuation and emoji
        5. Squeeze repeated spaces

    Example:
        "@sam I am SO happy!! http://x.co" -> "i am so happy"
    """
    text = str(text).lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = NON_LETTER_PATTERN.sub(" ", text)
    text = EXTRA_SPACE_PATTERN.sub(" ", text).strip()
    return text
