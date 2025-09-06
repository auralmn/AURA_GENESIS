import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# --- SBERT for sentence embeddings ---
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device="mps")

# --- 1. Load Dataset & Collect Unique Labels ---
dataset = []
labels_set = set()
intent_set = set()
tone_set = set()

for line in open('train.jsonl', 'r', encoding='utf-8'):
    line = line.strip()
    if line:
        record = json.loads(line)
        dataset.append(record)
        primary_label = record['plutchik']['primary']
        if isinstance(primary_label, list):
            primary_label = primary_label[0]
        labels_set.add(primary_label)
        intent = record.get("intent", None)
        if intent:
            intent_set.add(intent)
        tone = record.get("tone", None)
        if tone:
            tone_set.add(tone)

PLUTCHIK_LABELS = sorted(list(labels_set))
LABEL_TO_IDX = {e: i for i, e in enumerate(PLUTCHIK_LABELS)}
IDX_TO_LABEL = {i: e for e, i in LABEL_TO_IDX.items()}
INTENT_LABELS = sorted(list(intent_set))
INTENT_TO_IDX = {e: i for i, e in enumerate(INTENT_LABELS)}
IDX_TO_INTENT = {i: e for i, e in enumerate(INTENT_LABELS)}
TONE_LABELS = sorted(list(tone_set))
TONE_TO_IDX = {e: i for i, e in enumerate(TONE_LABELS)}
IDX_TO_TONE = {i: e for i, e in enumerate(TONE_LABELS)}

with open('emotion_labels.json', 'w', encoding='utf-8') as f:
    json.dump({
        'LABEL_TO_IDX': LABEL_TO_IDX,
        'IDX_TO_LABEL': IDX_TO_LABEL,
        'INTENT_TO_IDX': INTENT_TO_IDX,
        'IDX_TO_INTENT': IDX_TO_INTENT,
        'TONE_TO_IDX': TONE_TO_IDX,
        'IDX_TO_TONE': IDX_TO_TONE
    }, f)

# --- 2. Sinewave + SBERT Text Features ---
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

def multi_sine_embedding(record, length=32):
    primary = record['plutchik']['primary']
    if isinstance(primary, list):
        primary = primary[0]
    intensity = record['plutchik']['intensity']
    amp = PLUTCHIK_EMBEDDING_PARAMS[primary]['amp'] * intensity
    freq = PLUTCHIK_EMBEDDING_PARAMS[primary]['freq']
    phase = PLUTCHIK_EMBEDDING_PARAMS[primary]['phase']
    t = np.linspace(0, 2 * np.pi, length)
    emb = amp * np.sin(freq * t + phase)
    secondary = record['plutchik'].get('secondary')
    if secondary:
        if isinstance(secondary, list):
            secondary = secondary[0]
        if secondary in PLUTCHIK_EMBEDDING_PARAMS:
            params_sec = PLUTCHIK_EMBEDDING_PARAMS[secondary]
            emb += 0.5 * (params_sec['amp'] * intensity) * np.sin(params_sec['freq'] * t + params_sec['phase'])
    # ---- Add extra features ----
    text = record.get("text", "")
    extra = [
        len(text) / 100.0,
        int("!" in text),
        int(record.get("tone", "") in {"euphoric", "tense", "somber", "peaceful", "amazed"})
    ]
    # Append SBERT embedding for text
    text_emb = sbert_model.encode(text)
    return np.concatenate([emb, np.array(extra, dtype=np.float32), text_emb])

# --- 3. Shuffle & Split ---
random.seed(42)
indices = list(range(len(dataset)))
random.shuffle(indices)
split_ratio = 0.8
split_index = int(len(dataset) * split_ratio)
train_indices = indices[:split_index]
test_indices = indices[split_index:]
train_data = [dataset[i] for i in train_indices]
test_data = [dataset[i] for i in test_indices]

# --- 4. Feature & Label Extraction ---
def get_features_and_labels(data, label_map, embedding_func, label_field="primary"):
    X, y = [], []
    for record in data:
        emb = embedding_func(record)
        if label_field == "intent":
            label_val = record.get("intent", "none")
        elif label_field == "tone":
            label_val = record.get("tone", "none")
        else:
            label_val = record['plutchik'][label_field]
        if isinstance(label_val, list):
            label_val = label_val[0]
        label = label_map.get(label_val)
        if label is not None:
            X.append(emb)
            y.append(label)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(y)

X_train, y_train = get_features_and_labels(train_data, LABEL_TO_IDX, multi_sine_embedding)
X_test, y_test = get_features_and_labels(test_data, LABEL_TO_IDX, multi_sine_embedding)
X_intent_train, y_intent_train = get_features_and_labels(train_data, INTENT_TO_IDX, multi_sine_embedding, label_field="intent")
X_intent_test, y_intent_test = get_features_and_labels(test_data, INTENT_TO_IDX, multi_sine_embedding, label_field="intent")
X_tone_train, y_tone_train = get_features_and_labels(train_data, TONE_TO_IDX, multi_sine_embedding, label_field="tone")
X_tone_test, y_tone_test = get_features_and_labels(test_data, TONE_TO_IDX, multi_sine_embedding, label_field="tone")

print("X_train shape (emotion):", X_train.size())
print("X_train shape (intent):", X_intent_train.size())
print("X_train shape (tone):", X_tone_train.size())

# --- 5. PyTorch Classifier ---
class FeedforwardClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.model(x)

def train_classifier(model, X, y, lr=0.005, epochs=30, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_samples = X.size(0)
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            xb = X_shuffled[start:end]
            yb = y_shuffled[start:end]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            preds = model(X).argmax(dim=1)
            acc = (preds == y).float().mean().item()
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch} | Train Acc: {acc:0.3f}")

def eval_classifier(model, X, y):
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        accuracy = (preds == y).float().mean().item()
    return preds, accuracy

# --- 6. Train & Save ---
dim = X_train.size(1)
clf_emotion = FeedforwardClassifier(input_dim=dim, num_classes=len(LABEL_TO_IDX))
train_classifier(clf_emotion, X_train, y_train, lr=0.005, epochs=30)
torch.save(clf_emotion.state_dict(), 'clf_emotion.pt')

clf_intent = FeedforwardClassifier(input_dim=dim, num_classes=len(INTENT_TO_IDX))
train_classifier(clf_intent, X_intent_train, y_intent_train, lr=0.005, epochs=30)
torch.save(clf_intent.state_dict(), 'clf_intent.pt')

clf_tone = FeedforwardClassifier(input_dim=dim, num_classes=len(TONE_TO_IDX))
train_classifier(clf_tone, X_tone_train, y_tone_train, lr=0.005, epochs=30)
torch.save(clf_tone.state_dict(), 'clf_tone.pt')

print("Models saved!")

# --- 7. Load for Eval ---
clf_emotion2 = FeedforwardClassifier(input_dim=dim, num_classes=len(LABEL_TO_IDX))
clf_emotion2.load_state_dict(torch.load('clf_emotion.pt'))
clf_intent2 = FeedforwardClassifier(input_dim=dim, num_classes=len(INTENT_TO_IDX))
clf_intent2.load_state_dict(torch.load('clf_intent.pt'))
clf_tone2 = FeedforwardClassifier(input_dim=dim, num_classes=len(TONE_TO_IDX))
clf_tone2.load_state_dict(torch.load('clf_tone.pt'))
print("Models loaded!")

# --- 8. Evaluate & Print Results ---
preds_emotion, acc_emotion = eval_classifier(clf_emotion2, X_test, y_test)
preds_intent, acc_intent = eval_classifier(clf_intent2, X_intent_test, y_intent_test)
preds_tone, acc_tone = eval_classifier(clf_tone2, X_tone_test, y_tone_test)

print("Validation accuracy (emotion):", acc_emotion)
print("Validation accuracy (intent):", acc_intent)
print("Validation accuracy (tone):", acc_tone)

for i in range(10):
    print("-" * 40)
    print(f"Text: {test_data[i]['text']}")
    print(f"True emotion: {IDX_TO_LABEL.get(y_test[i].item(), 'UNKNOWN')} | Predicted: {IDX_TO_LABEL.get(preds_emotion[i].item(), 'UNKNOWN')}")
    print(f"True intent: {IDX_TO_INTENT.get(y_intent_test[i].item(), 'UNKNOWN')} | Predicted: {IDX_TO_INTENT.get(preds_intent[i].item(), 'UNKNOWN')}")
    print(f"True tone: {IDX_TO_TONE.get(y_tone_test[i].item(), 'UNKNOWN')} | Predicted: {IDX_TO_TONE.get(preds_tone[i].item(), 'UNKNOWN')}\n")
