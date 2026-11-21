Individual Defense Guide

1. Explain the task in 20 seconds

I built a binary text classifier that receives one Kazakh customer review or comment and predicts whether profanity or insulting language is present.

* Label 0 means no profanity.
* Label 1 means profanity is present.

The project focuses on Kazakh and Russian profanity, including mixed-language comments, recognizable Latin transliterations, and spelling variations.

2. Explain the data

* I prepared and manually labeled a subset of 3,061 real reviews/comments associated with the KazSAnDRA dataset.
* The original KazSAnDRA labels represent sentiment or ratings, not profanity. I therefore created task-specific binary labels.
* The dataset contains 1,503 clean comments and 1,558 profane comments.
* No synthetic examples were added.
* I manually reviewed suspicious annotations and corrected 23 labels.
* The annotation process was performed by one annotator. Inter-annotator agreement was not measured.

The dataset was divided into three subsets:

Split	Samples
Train	2,525
Validation	265
Test	271
Total	3,061

The split was retained for the final experiment.

Data leakage checks

I checked whether identical or highly similar comments appeared in different splits.

* Exact normalized cross-split duplicates: 0.
* Character TF-IDF near-duplicate pairs with cosine similarity >= 0.90: 0.
* At similarity >= 0.85, the audit found 1 Train–Validation pair, 3 Train–Test pairs, and 0 Validation–Test pairs.

These checks reduce the risk of duplicate-related data leakage, but they cannot guarantee that all semantically similar comments have been identified.

3. Explain the model

I chose word and character TF-IDF features with Logistic Regression.

Word n-grams (1–2) learn individual words and short expressions.

For example, the model can learn that certain words or two-word combinations frequently appear in profane comments.

Character n-grams (2–5) learn short sequences of characters. They help with:

* Spelling errors.
* Mixed Cyrillic and Latin scripts.
* Elongated letters.
* Some transliteration variants.
* Slightly modified profanity words.

Logistic Regression learns a binary decision boundary using the extracted text features. It calculates a score that the model uses to classify a comment.

I selected this approach because it is lightweight, relatively fast, and suitable as a baseline for a dataset of this size.

4. Trace one comment through the system

For example, consider this input:

Кымбаттап кетипти говно

The prediction process works as follows:

1. The raw comment is passed to the saved model pipeline.
2. The word TF-IDF vectorizer extracts word and word-pair features.
3. The character TF-IDF vectorizer extracts character sequences of lengths 2 to 5.
4. The two feature sets are combined.
5. Logistic Regression calculates the profanity score.
6. The score is compared with the selected decision threshold: 0.44526553579034467.
7. The program returns label 0 or 1, the estimated probability score, and the threshold.

If the score is greater than or equal to the threshold, the comment is classified as containing profanity.

Otherwise, it is classified as having no detected profanity.

5. Explain training and validation

* Random seed: 42.
* The vectorizers and classifier are fitted only on the training data.
* I compared five values of C: 0.1, 0.3, 1, 3, and 10.
* The models were evaluated on the validation set.
* I selected C=3.
* The decision threshold was selected to maximize the F1 score for the profanity class on Validation.
* The final selected threshold is 0.44526553579034467, approximately 0.4453.
* The held-out Test set was used for final evaluation after model and threshold selection.

What does C control?

C is the inverse of regularization strength in Logistic Regression.

* Smaller C means stronger regularization.
* Larger C means weaker regularization.

Regularization helps control model complexity and reduce overfitting.

Why use a separate validation set?

The validation set helps select model parameters and the decision threshold without using the final test data.

The test set is reserved for evaluating the selected model on unseen examples.

6. Explain final results

The final model was evaluated on 271 held-out test comments.

Metric	Result
Accuracy	88.93%
Profanity precision	92.36%
Profanity recall	88.96%
Profanity F1	90.63%

Confusion matrix

	Predicted 0	Predicted 1
True 0	96	12
True 1	18	145

This means:

* True Negative (96): clean comments correctly classified as clean.
* False Positive (12): clean comments incorrectly classified as profane.
* False Negative (18): profane comments incorrectly classified as clean.
* True Positive (145): profane comments correctly detected.

What do the metrics mean?

Accuracy measures the proportion of all correctly classified comments.

Precision answers: among comments predicted as profane, how many were actually profane?

Recall answers: among all truly profane comments, how many did the model detect?

F1-score is the harmonic mean of precision and recall. It balances both metrics.

How should we interpret the results?

The model achieved 90.63% F1 for the profanity class on the held-out test set.

Precision is slightly higher than recall. This means that when the model predicts profanity, it is usually correct, although it still misses some profane comments.

There are 12 false positives and 18 false negatives. The balance between these errors matters for moderation systems.

7. Explain model limitations and mistakes

The model made 30 errors on the held-out test set:

* 12 false positives.
* 18 false negatives.

False negatives

A false negative happens when a profane comment is classified as clean.

Examples from the test set:

* Маған ұнайды бляя😎😎 — the model missed a short comment containing an explicit profanity word.
* A comment written in Latin transliteration — the model failed to recognize a less common spelling pattern.
* Long comments containing profanity together with mostly neutral text — the profanity signal may be less dominant.

These examples show that the model has difficulties with rare spelling variations, transliteration, and long mixed-content comments.

False positives

A false positive happens when a clean comment is classified as profane.

Examples from the test set:

* Маңғыстауда мал жоқ па? Дым шықпайды ғой — the word мал can refer to livestock rather than an insult.
* Код неге келмейді... — a negative complaint without clear profanity.
* Сбоит синхронизация с amazfit bip. — a technical complaint incorrectly classified as profane.

These examples show that the model can confuse negative language or context-dependent words with profanity.

Main limitations

1. Context understanding: TF-IDF learns word and character patterns but does not fully understand sentence meaning.
2. Spelling variation: Unusual transliterations and non-standard spellings can be difficult to recognize.
3. Long comments: Profanity can be mixed with neutral text, making classification more difficult.
4. Dataset size: The model was trained on a relatively small dataset of 3,061 comments.

A false positive can incorrectly flag an acceptable comment. A false negative can allow profanity through moderation.

The preferred balance between precision and recall depends on the moderation policy. Future work could include collecting more naturally occurring examples, improving annotation quality, and comparing the current model with multilingual transformer architectures.

8. Limitations and future work

The project has several limitations:

* The dataset contains only 3,061 manually curated comments.
* The annotation process was performed by one annotator.
* Inter-annotator agreement was not measured.
* The dataset may not represent the natural prevalence of profanity in real-world reviews.
* The model cannot fully understand context, sarcasm, or complex sentence meaning.
* Rare transliterations and creative spelling variations remain challenging.
* The probability score is not guaranteed to be calibrated.

Possible future improvements:

1. Collect a larger and more diverse dataset of naturally occurring Kazakh comments.
2. Use multiple annotators and measure inter-annotator agreement.
3. Add more examples of spelling variations and transliteration through real data collection and manual annotation.
4. Compare the current baseline with multilingual transformer models.
5. Evaluate probability calibration and different decision thresholds.
6. Test the model on an independently collected dataset before deployment.

9. Likely defense questions

Why did you choose binary classification?

Because the goal is to detect whether a comment contains profanity or insulting language. The task has two possible classes: clean and profane.

Why not use sentiment labels directly?

Because sentiment and profanity are different tasks. A negative review can be clean, and a positive review can still contain profanity.

Why use character n-grams?

Profanity is often misspelled, elongated, masked, or transliterated. Character patterns can capture some of these variations.

Why use Logistic Regression?

It is lightweight, fast, interpretable compared with more complex models, and works well with sparse TF-IDF features.

What is TF-IDF?

TF-IDF stands for Term Frequency–Inverse Document Frequency. It represents text numerically by assigning weights to words or character sequences based on their importance in the documents.

What is an n-gram?

An n-gram is a sequence of n consecutive units.

For word n-grams:

* Unigram: one word.
* Bigram: two consecutive words.

For character n-grams, the units are characters rather than words.

What is overfitting?

Overfitting happens when a model learns details specific to the training data and performs worse on unseen data. Separate training, validation, and test sets help evaluate generalization.

What does C control?

C is the inverse regularization strength in Logistic Regression. Smaller C means stronger regularization, while larger C means weaker regularization.

Why use a threshold instead of simply accepting the default prediction?

The threshold controls how high the profanity score must be before the model predicts class 1. We selected it using validation data to maximize positive-class F1.

Why is the threshold approximately 0.4453 instead of 0.5?

Because the threshold was selected using validation data. The value that maximized positive-class F1 was approximately 0.4453, rather than the default 0.5.

Can the model understand the meaning of мал?

Not reliably. TF-IDF mostly learns word and character patterns, not deep contextual meaning. That is why literal uses can become false positives.

Why is accuracy not enough?

Accuracy does not show the balance between missed profanity and incorrectly flagged clean comments. Precision, recall, F1, and the confusion matrix provide a more complete evaluation.

What is the difference between false positive and false negative?

A false positive is a clean comment incorrectly classified as profane.

A false negative is a profane comment incorrectly classified as clean.

Did you use synthetic data?

No. The project uses real review/comment texts. No synthetic examples were added.

What is the main contribution of your project?

The main contribution is preparing and manually labeling a Kazakh profanity detection dataset, developing a reproducible binary classification pipeline, and evaluating it using a separate held-out test set.
