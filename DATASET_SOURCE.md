# Dataset Source and Preparation

## 1. Original Dataset

This project uses a manually selected subset of the KazSAnDRA review dataset.

Sources:
- KazSAnDRA project: https://issai.nu.edu.kz/kazsandra/
- Official repository: https://github.com/IS2AI/KazSAnDRA
- Dataset paper: Yeshpanov & Varol (2024), LREC-COLING: https://aclanthology.org/2024.lrec-main.844/

KazSAnDRA was developed for sentiment and attitude analysis. Its original sentiment labels were not used as profanity labels in this project.

## 2. Dataset Preparation

A subset of 3,061 review/comment records was selected and manually labeled for binary profanity detection.

- `0` — no profanity detected.
- `1` — profanity or insulting language detected.

No synthetic examples were added. The processed dataset retains source review IDs for traceability. The text was additionally cleaned to replace detected social-media handles with `[SOCIAL_HANDLE]`. This is a targeted automated cleanup, not a formal anonymization assessment.

## 3. Annotation Guidelines

The annotation policy treats recognizable Kazakh and Russian profanity as label `1`, including mixed-language comments, recognizable Latin transliterations, clear spelling variations, elongated forms, and unambiguously masked profanity. Negative opinions without profanity are label `0`. Words that resemble profanity but have a literal meaning in context are label `0`. Ambiguous cases are assessed in context rather than by keyword matching alone.

For example, `мал` may refer literally to livestock or function as an insult; its label depends on context.

## 4. Manual Label Review

The initial annotations were manually inspected, and 23 labels were corrected after reviewing suspicious cases. The annotation process was performed by one annotator. Inter-annotator agreement was not measured.

## 5. Data Splitting and Leakage Prevention

The final dataset contains:
- Train: 2,525
- Validation: 265
- Test: 271

Cross-split duplicate checks were performed after lowercasing, punctuation/symbol removal, and whitespace normalization. No exact normalized duplicates were found. Character TF-IDF near-duplicate checks using character n-grams (2–5) found no cross-split pairs with cosine similarity of 0.90 or higher.

These checks reduce duplicate-related leakage risk but cannot guarantee the absence of all semantically similar comments. Train is used to fit the vectorizers and classifier; Validation is used for hyperparameter and threshold selection; Test is reserved for final evaluation.

## 6. Data Limitations

The dataset is manually curated and may not represent the natural prevalence or full diversity of profanity in real customer reviews. The annotation process relies on one annotator, and some expressions require contextual interpretation. The dataset should be considered a task-specific research subset rather than a comprehensive collection of Kazakh profanity.

## 7. Attribution and Licensing

The original KazSAnDRA dataset is attributed to Rustem Yeshpanov and Huseyin Atakan Varol. Consult the official project page and repository for the license and redistribution terms applicable to the specific dataset files:
- https://issai.nu.edu.kz/kazsandra/
- https://github.com/IS2AI/KazSAnDRA

The license of the research paper should not automatically be assumed to apply to the dataset files. Before redistributing the review text, verify the applicable dataset license and retain the required attribution. This project is an independent academic work and does not imply endorsement by the original authors.
