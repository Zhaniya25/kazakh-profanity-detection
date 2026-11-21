"""Predict whether a Kazakh/Russian mixed review contains profanity."""
from pathlib import Path
import argparse, json
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'kazakh_profanity_tfidf_logreg.joblib'
CONFIG_PATH = ROOT / 'models' / 'config.json'

def predict_text(text, model_path=MODEL_PATH, config_path=CONFIG_PATH):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Input text must be a non-empty string.')
    model = joblib.load(model_path)
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    probability = float(model.predict_proba([text])[0, 1])
    threshold = float(config['threshold'])
    label = int(probability >= threshold)
    return {'label': label, 'prediction': 'profanity present' if label else 'no profanity detected',
            'profanity_probability': round(probability, 4), 'threshold': round(threshold, 4)}

def main():
    parser = argparse.ArgumentParser(description='Kazakh profanity detection')
    parser.add_argument('--text', required=True, help='Review or comment to classify')
    args = parser.parse_args()
    print(json.dumps(predict_text(args.text), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
