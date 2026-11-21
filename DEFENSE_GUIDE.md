# Individual Defense Guide

## 1. Explain the task in 20 seconds

I built a binary text classifier that receives one Kazakh customer review/comment and predicts whether profanity or insulting language is present. Label 0 means no profanity; Label 1 means profanity is present. Mixed Kazakh–Russian comments and recognizable Latin transliterations are included in the positive class.

## 2. Explain the data

- I prepared and manually labeled a 3,061-review subset associated with KazSAnDRA.
- The original KazSAnDRA labels represent sentiment/ratings, not profanity. I therefore created task-specific binary labels and manually audited suspicious cases.
- Final classes: 1,503 clean and 1,558 profane.
- No synthetic examples were added.
- Split: 2,525 train, 265 validation, 271 test.
- I moved three confirmed near-duplicate validation records into train. After this, there were no exact normalized duplicates or character-TF-IDF cross-split pairs at similarity >= 0.90.

## 3. Explain the model

I chose word and character TF-IDF features with Logistic Regression.

- Word n-grams (1–2) learn words and short expressions.
- Character n-grams (2–5) help with spelling errors, mixed Cyrillic/Latin text, elongated letters, and some transliteration.
- Logistic Regression learns a binary decision boundary and returns a score that is converted into a probability-like value.
- The model is small, fast, and suitable as a baseline for a dataset of this size.

## 4. Trace one comment through the system

1. The raw comment is passed to the saved pipeline.
2. The word TF-IDF vectorizer creates word and word-pair features.
3. The character TF-IDF vectorizer creates character sequences of lengths 2 to 5.
4. The two sparse feature sets are combined.
5. Logistic Regression calculates the profanity score.
6. The score is compared with the validation-selected threshold 0.4458.
7. The program returns label 0 or 1, the score, and the threshold.

## 5. Explain training and validation

- Random seed: 42.
- The vectorizers are fitted only on training text.
- I compared C values 0.1, 0.3, 1, 3, and 10 using validation data.
- C=3 was selected. The decision threshold 0.4458 was selected to maximize profanity-class F1 on validation.
- The held-out test set was used only after the model and threshold were selected.

## 6. Explain final results

On 271 test examples:

- Accuracy: 88.56%.
- Profanity precision: 92.31%.
- Profanity recall: 88.34%.
- Profanity F1: 90.28%.
- Confusion matrix: TN=96, FP=12, FN=19, TP=144.

Precision answers: among comments predicted as profane, how many were actually profane?
Recall answers: among all truly profane comments, how many did the model detect?
F1 is the harmonic mean of precision and recall.

## 7. Explain mistakes

- False positive: “Маңғыстауда мал жоқ па?” uses “мал” literally to mean livestock. The model over-associates this token with insults.
- False negative: “Маған ұнайды бляя” uses an elongated spelling variant that is rare in training.
- False negative: Latin transliteration with diacritics is difficult for a character vocabulary trained on a limited sample.
- False negative: a long mostly neutral review contains one profanity token, whose signal can be diluted.

A false positive can incorrectly hide an acceptable review. A false negative can allow profanity through moderation. The correct trade-off depends on the moderation policy.

## 8. Limitations and future work

The dataset is relatively small, manually curated, and close to balanced by design. It does not reflect real-world profanity prevalence. The model does not fully understand context, sarcasm, or all spelling variants. The probability score is not guaranteed to be calibrated. Future work could add more naturally occurring, manually labeled examples and compare a multilingual transformer while retaining this baseline.

## 9. Likely defense questions

**Why not use sentiment labels directly?**  
Because sentiment and profanity are different tasks. A negative review can be clean, and a positive review can still contain profanity.

**Why use character n-grams?**  
Profanity is often misspelled, elongated, masked, or transliterated. Character patterns can capture some of these variants.

**Why use validation data?**  
To choose model settings and the decision threshold without contaminating the final test evaluation.

**What is overfitting?**  
When a model learns details specific to training data and performs worse on unseen data. The separate validation and test sets help detect and measure generalization.

**What does C control?**  
C is the inverse regularization strength in Logistic Regression. A smaller C applies stronger regularization; a larger C allows a more flexible fit.

**Can the model understand the meaning of “мал”?**  
Not reliably. TF-IDF mostly learns word and character patterns, not deep contextual meaning. That is why literal uses can become false positives.

**Why is accuracy not enough?**  
Because it does not show the balance between missed profanity and incorrectly flagged clean comments. Precision, recall, F1, and the confusion matrix give a more complete picture.
