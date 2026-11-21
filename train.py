"""Train a Kazakh profanity classifier and evaluate once on the held-out test set."""
from pathlib import Path
import argparse, json, random
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, precision_recall_curve)

SEED = 42
C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

def make_model(C):
    features = FeatureUnion([
        ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2,
                                 max_features=50000, sublinear_tf=True,
                                 token_pattern=r'(?u)\b\w+\b')),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=2,
                                 max_features=150000, sublinear_tf=True))
    ])
    return Pipeline([
        ('features', features),
        ('classifier', LogisticRegression(C=C, max_iter=3000, solver='liblinear', random_state=SEED))
    ])

def select_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best]), float(f1[best])

def main(data_path, output_dir):
    set_seed()
    data = pd.read_csv(data_path)
    required = {'text', 'label', 'split'}
    if not required.issubset(data.columns):
        raise ValueError(f'Dataset must contain columns: {required}')
    if data['text'].isna().any() or data['label'].isna().any():
        raise ValueError('Missing text or labels found.')
    if not set(data['label'].unique()).issubset({0, 1}):
        raise ValueError('Labels must be binary 0/1.')
    if set(data['split'].unique()) != {'train', 'val', 'test'}:
        raise ValueError('Expected train, val, and test splits.')

    train = data[data.split == 'train']
    val = data[data.split == 'val']
    test = data[data.split == 'test']
    X_train, y_train = train.text.astype(str), train.label.astype(int)
    X_val, y_val = val.text.astype(str), val.label.astype(int)
    X_test, y_test = test.text.astype(str), test.label.astype(int)

    candidates = []
    fitted = {}
    for C in C_GRID:
        model = make_model(C)
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        threshold, val_f1 = select_threshold(y_val, val_prob)
        val_pred = (val_prob >= threshold).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_val, val_pred, average='binary', zero_division=0)
        candidates.append({'C': C, 'threshold': threshold, 'val_precision': float(p),
                           'val_recall': float(r), 'val_f1': float(f)})
        fitted[C] = model
    best = sorted(candidates, key=lambda x: (x['val_f1'], x['val_recall'], x['val_precision'], -x['C']), reverse=True)[0]
    model, threshold = fitted[best['C']], best['threshold']

    # Test set is touched only after model and threshold selection are complete.
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    p, r, f, _ = precision_recall_fscore_support(y_test, test_pred, average='binary', zero_division=0)
    metrics = {'test_size': int(len(y_test)), 'accuracy': float(accuracy_score(y_test, test_pred)),
               'profanity_precision': float(p), 'profanity_recall': float(r), 'profanity_f1': float(f),
               'confusion_matrix': cm.tolist(), 'selected_C': best['C'], 'threshold': threshold,
               'validation_candidates': candidates, 'sklearn_version': sklearn.__version__, 'seed': SEED}

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / 'kazakh_profanity_tfidf_logreg.joblib')
    config = {'model': 'word + character TF-IDF with Logistic Regression', 'seed': SEED,
              'selected_C': best['C'], 'threshold': threshold, 'C_grid': C_GRID,
              'train_size': len(train), 'validation_size': len(val), 'test_size': len(test),
              'sklearn_version': sklearn.__version__}
    (out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'test_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame({'text': X_test.values, 'true_label': y_test.values, 'predicted_label': test_pred,
                  'profanity_probability': test_prob, 'correct': test_pred == y_test.values}).to_csv(out / 'test_predictions.csv', index=False)
    print('Validation model selection:')
    print(pd.DataFrame(candidates).to_string(index=False))
    print('\nFinal test results (evaluated once after selection):')
    print(classification_report(y_test, test_pred, digits=4, zero_division=0))
    print('Confusion matrix [[TN, FP], [FN, TP]]:')
    print(cm)
    print(f"Saved model and results to: {out.resolve()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/kazakh_profanity_dataset.csv')
    parser.add_argument('--output', default='models')
    args = parser.parse_args()
    main(args.data, args.output)
