# Error Analysis

The refreshed model made **30 errors** on the held-out test set of 271 examples.

- False positives (true 0, predicted 1): 12
- False negatives (true 1, predicted 0): 18

The full set of misclassified examples, predicted probabilities, and labels is available in `error_analysis.csv`.

Manual review is required before assigning linguistic causes to individual errors. The model's errors may involve context-dependent words, spelling variation, transliteration, masking, or long mixed-language comments; these explanations should only be stated for examples confirmed by a human reviewer.
