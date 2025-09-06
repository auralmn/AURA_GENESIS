import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer

# --- 1. Load SVC dataset from file ---
def load_svc_dataset(filename):
    data = []
    domains = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            data.append(record)
            domains.add(record['metadata']['domain'])
    return data, sorted(list(domains))

svc_data, domains = load_svc_dataset('train_svc.jsonl')
DOMAIN_TO_IDX = {d: i for i, d in enumerate(domains)}
IDX_TO_DOMAIN = {i: d for i, d in enumerate(domains)}

with open('svc_domain_labels.json', 'w', encoding='utf-8') as f:
    json.dump({"DOMAIN_TO_IDX": DOMAIN_TO_IDX, "IDX_TO_DOMAIN": IDX_TO_DOMAIN}, f)

# --- 2. Build SBERT & SVC Embeddings ---
sbert = SentenceTransformer('all-MiniLM-L6-v2')

def get_svc_embedding(record, sbert_model):
    svc = record['metadata']['svc']
    subj_emb = sbert_model.encode(svc['subject'])
    verb_emb = sbert_model.encode(svc['verb'])
    comp_emb = sbert_model.encode(svc['complement'])
    return np.concatenate([subj_emb, verb_emb, comp_emb])

def get_full_knowledge_embedding(record, sbert_model):
    text_emb = sbert_model.encode(record['text'])
    svc_emb = get_svc_embedding(record, sbert_model)
    domain = record['metadata']['domain']
    diff = record['metadata']['difficulty']
    domain_onehot = np.zeros(len(DOMAIN_TO_IDX), dtype=np.float32)
    domain_onehot[DOMAIN_TO_IDX[domain]] = 1
    return np.concatenate([text_emb, svc_emb, domain_onehot, [diff]])

X = np.array([get_full_knowledge_embedding(rec, sbert) for rec in svc_data])
y_domain = np.array([DOMAIN_TO_IDX[rec['metadata']['domain']] for rec in svc_data])
y_diff = np.array([rec['metadata']['difficulty'] for rec in svc_data])

X = torch.tensor(X, dtype=torch.float32)
y_domain = torch.tensor(y_domain, dtype=torch.long)
y_diff = torch.tensor(y_diff, dtype=torch.float32)

input_dim = X.size(1)
num_domains = len(domains)

# --- 3. Dataset Split ---
split_ratio = 0.8
idxs = np.arange(len(svc_data))
np.random.seed(42)
np.random.shuffle(idxs)
split_idx = int(len(idxs) * split_ratio)
train_idx = idxs[:split_idx]
test_idx = idxs[split_idx:]

X_train = X[train_idx]
y_domain_train = y_domain[train_idx]
y_diff_train = y_diff[train_idx]
X_test = X[test_idx]
y_domain_test = y_domain[test_idx]
y_diff_test = y_diff[test_idx]
test_data = [svc_data[i] for i in test_idx]

# --- 4. Classifier and Regressor Classes ---
class DomainClassifier(nn.Module):
    def __init__(self, input_dim, num_domains):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_domains)
        )
    def forward(self, x):
        return self.model(x)

class DifficultyRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x).squeeze(-1)

# --- 5. Train Domain Classifier ---
clf_domain = DomainClassifier(input_dim=input_dim, num_domains=num_domains)
optimizer = torch.optim.Adam(clf_domain.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    optimizer.zero_grad()
    logits = clf_domain(X_train)
    loss = criterion(logits, y_domain_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0 or epoch == 49:
        preds = logits.argmax(dim=1)
        acc = (preds == y_domain_train).float().mean().item()
        print(f"Epoch {epoch} | Train Domain Acc: {acc:.3f} | Loss: {loss.item():.3f}")

torch.save(clf_domain.state_dict(), 'svc_domain_classifier.pt')

# --- 6. Train Difficulty Regressor ---
regressor = DifficultyRegressor(input_dim=input_dim)
optimizer_r = torch.optim.Adam(regressor.parameters(), lr=0.01)
criterion_r = nn.MSELoss()

for epoch in range(50):
    optimizer_r.zero_grad()
    output = regressor(X_train)
    loss_r = criterion_r(output, y_diff_train)
    loss_r.backward()
    optimizer_r.step()
    if epoch % 10 == 0 or epoch == 49:
        print(f"Epoch {epoch} | Train Difficulty Loss: {loss_r.item():.4f}")

torch.save(regressor.state_dict(), 'svc_difficulty_regressor.pt')

print("Models and domain mappings saved.")

# --- 7. Reload Models and Mappings ---
def load_domain_classifier(input_dim, num_domains):
    clf = DomainClassifier(input_dim, num_domains)
    clf.load_state_dict(torch.load('svc_domain_classifier.pt'))
    clf.eval()
    return clf

def load_difficulty_regressor(input_dim):
    reg = DifficultyRegressor(input_dim)
    reg.load_state_dict(torch.load('svc_difficulty_regressor.pt'))
    reg.eval()
    return reg

def load_domain_mappings(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    # Make sure keys are integers
    return {int(k): v for k, v in mappings['IDX_TO_DOMAIN'].items()}

# --- 8. Evaluate & Print Predictions ---
clf_domain2 = load_domain_classifier(input_dim, num_domains)
regressor2 = load_difficulty_regressor(input_dim)
IDX_TO_DOMAIN = load_domain_mappings('svc_domain_labels.json')

with torch.no_grad():
    logits = clf_domain2(X_test)
    preds = logits.argmax(dim=1)
    output_diff = regressor2(X_test)

for i, rec in enumerate(test_data):
    svc = rec['metadata']['svc']
    print("-" * 40)
    print(f"Text: {rec['text']}")
    print(f"SVC: Subject: {svc['subject']} | Verb: {svc['verb']} | Complement: {svc['complement']}")
    print(f"True domain: {rec['metadata']['domain']} | Pred domain: {IDX_TO_DOMAIN.get(preds[i].item(), 'UNKNOWN')}")
    print(f"True difficulty: {rec['metadata']['difficulty']:.2f} | Pred difficulty: {float(output_diff[i]):.2f}\n")
    if i > 10:
        exit()