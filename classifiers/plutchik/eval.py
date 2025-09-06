import numpy as np
import json

# --- 1. Load label mappings ---
with open('emotion_classifier_labels.json', 'r', encoding='utf-8') as f:
    labels_dict = json.load(f)
LABEL_TO_IDX = labels_dict['LABEL_TO_IDX']
IDX_TO_LABEL = {int(k): v for k, v in labels_dict['IDX_TO_LABEL'].items()}

# --- 2. Load model weights ---
W = np.load('emotion_classifier_W.npy')
b = np.load('emotion_classifier_b.npy')

# --- 3. Generate Embedding Params for All Labels ---
def generate_default_params(labels):
    base_freq = 1.5
    base_phase = 0.5
    params = {}
    for idx, label in enumerate(labels):
        params[label] = {
            "freq": base_freq + 0.3 * idx,
            "amp": 0.7,
            "phase": base_phase + 0.4 * idx
        }
    return params

PLUTCHIK_LABELS = sorted(list(LABEL_TO_IDX.keys()), key=lambda x: LABEL_TO_IDX[x])
PLUTCHIK_EMBEDDING_PARAMS = generate_default_params(PLUTCHIK_LABELS)

# --- 4. Embedding functions ---
def sinewave_embedding(primary, intensity, length=32):
    params = PLUTCHIK_EMBEDDING_PARAMS.get(primary)
    if not params: return np.zeros(length)
    amp = params["amp"] * intensity
    freq = params["freq"]
    phase = params["phase"]
    t = np.linspace(0, 2 * np.pi, length)
    return amp * np.sin(freq * t + phase)

def multi_sine_embedding(record, length=32):
    primary = record['plutchik']['primary']
    # Handle list or str
    if isinstance(primary, list):
        primary = primary[0]
    intensity = record['plutchik']['intensity']
    emb = sinewave_embedding(primary, intensity, length)
    secondary = record['plutchik'].get('secondary')
    if secondary:
        if isinstance(secondary, list):
            secondary = secondary[0]
        if secondary in PLUTCHIK_EMBEDDING_PARAMS:
            emb += 0.5 * sinewave_embedding(secondary, intensity, length)
    return emb

# --- 5. Emotion Classifier definition ---
class EmotionClassifier:
    def __init__(self, input_dim, num_classes):
        self.W = np.zeros((input_dim, num_classes))
        self.b = np.zeros(num_classes)

    def load(self, W, b):
        self.W = W
        self.b = b

    def softmax(self, z):
        expz = np.exp(z - np.max(z, axis=1, keepdims=True))
        return expz / np.sum(expz, axis=1, keepdims=True)

    def predict(self, X):
        logits = X @ self.W + self.b
        return self.softmax(logits)

# --- 6. Load test dataset ---
test_dataset = []
with open('amygdala_test2.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            record = json.loads(line)
            test_dataset.append(record)

# --- 7. Prepare test embeddings/labels ---
X_test = []
y_test = []

for record in test_dataset:
    emb = multi_sine_embedding(record)
    label = LABEL_TO_IDX.get(record['plutchik']['primary'])
    X_test.append(emb)
    y_test.append(label)
X_test = np.array(X_test)
y_test = np.array(y_test)

# --- 8. Run inference ---
clf = EmotionClassifier(input_dim=X_test.shape[1], num_classes=len(LABEL_TO_IDX))
clf.load(W, b)

probs = clf.predict(X_test)
preds = np.argmax(probs, axis=1)
accuracy = np.mean(preds == y_test)

print(f"\nEval accuracy: {accuracy:0.3f}\n")

for i, record in enumerate(test_dataset):
    true_label = IDX_TO_LABEL[y_test[i]]
    pred_label = IDX_TO_LABEL[preds[i]]
    print(f"ID: {record['id']}")
    print(f"Text: {record['text']}")
    print(f"True: {true_label} | Pred: {pred_label} | Prob: {probs[i][preds[i]]:0.2f}")
    print(f"Intensity: {record['plutchik']['intensity']}   Tone: {record.get('tone','')}")
    print("---")

