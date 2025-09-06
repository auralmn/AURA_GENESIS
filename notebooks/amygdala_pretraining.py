import numpy as np
import json

# --- Step 1: Load Dataset & Find Unique Labels ---
dataset = []
labels_set = set()

with open('train.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line:
            record = json.loads(line)
            dataset.append(record)
            primary_label = record['plutchik']['primary']
            print(primary_label)
            try:

                labels_set.add(primary_label)
            except Exception as e:
                continue

PLUTCHIK_LABELS = sorted(list(labels_set))
LABEL_TO_IDX = {e: i for i, e in enumerate(PLUTCHIK_LABELS)}
IDX_TO_LABEL = {i: e for e, i in LABEL_TO_IDX.items()}

print("Unique emotion labels found:", PLUTCHIK_LABELS)

# --- Step 2: Generate Sinewave Embedding Parameters for All Labels ---

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

PLUTCHIK_EMBEDDING_PARAMS = generate_default_params(PLUTCHIK_LABELS)

# --- Step 3: Convert Dataset to Sinewave Embeddings/Labels ---

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
    # Handle if primary is a list
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

X = []
y = []

for record in dataset:
    emb = multi_sine_embedding(record)
    primary = record['plutchik']['primary']
    # Handle if primary is a list
    if isinstance(primary, list):
        primary = primary[0]
    label = LABEL_TO_IDX.get(primary)
    X.append(emb)
    y.append(label)
X = np.array(X)
y = np.array(y)

print("Shape of X:", X.shape)
print("Sample embedding first record:", X[0])

# --- Step 4: Emotion Classifier (NumPy Softmax NN) ---

class EmotionClassifier:
    def __init__(self, input_dim, num_classes):
        self.W = np.random.randn(input_dim, num_classes) * 0.1
        self.b = np.zeros(num_classes)

    def softmax(self, z):
        expz = np.exp(z - np.max(z, axis=1, keepdims=True))
        return expz / np.sum(expz, axis=1, keepdims=True)

    def predict(self, X):
        logits = X @ self.W + self.b
        return self.softmax(logits)

    def train(self, X, y, lr=0.09, epochs=400):
        for epoch in range(epochs):
            logits = X @ self.W + self.b
            probs = self.softmax(logits)
            y_onehot = np.eye(self.b.shape[0])[y]
            grad_W = X.T @ (probs - y_onehot) / len(X)
            grad_b = np.sum(probs - y_onehot, axis=0) / len(X)
            self.W -= lr * grad_W
            self.b -= lr * grad_b
            if epoch % 100 == 0:
                acc = np.mean(np.argmax(probs, axis=1) == y)
                print(f"Epoch {epoch} | Train Acc: {acc:0.3f}")

# --- Step 5: Train and Test ---


def save(clf):
    np.save('emotion_classifier_W.npy', clf.W)
    np.save('emotion_classifier_b.npy', clf.b)

def load():
        # Load weights and biases
    W = np.load('emotion_classifier_W.npy')
    b = np.load('emotion_classifier_b.npy')

    # Load label mappings
    with open('emotion_classifier_labels.json', 'r', encoding='utf-8') as f:
        labels_dict = json.load(f)
        LABEL_TO_IDX = labels_dict['LABEL_TO_IDX']
        IDX_TO_LABEL = {int(k): v for k, v in labels_dict['IDX_TO_LABEL'].items()}  # Convert keys to int if needed

        # Re-instantiate the classifier
        clf = EmotionClassifier(input_dim=W.shape[0], num_classes=W.shape[1])
        clf.W = W
        clf.b = b
    return clf


# Save label mappings as JSON
with open('emotion_classifier_labels.json', 'w', encoding='utf-8') as f:
    json.dump({"LABEL_TO_IDX": LABEL_TO_IDX, "IDX_TO_LABEL": IDX_TO_LABEL}, f)

clf = EmotionClassifier(input_dim=X.shape[1], num_classes=len(PLUTCHIK_LABELS))
clf.train(X, y, lr=0.09, epochs=400)

save(clf)


probs = clf.predict(X)
preds = np.argmax(probs, axis=1)
accuracy = np.mean(preds == y)
print("Final train accuracy:", accuracy)
for i in range(min(len(dataset), 20)):
    print(f"Text: {dataset[i]['text']}\nTrue: {IDX_TO_LABEL[y[i]]} | Pred: {IDX_TO_LABEL[preds[i]]}\n")
