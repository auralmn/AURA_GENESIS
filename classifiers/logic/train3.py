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
    realms = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            data.append(record)
            domains.add(record['metadata']['domain'])
            realms.add(record['realm'])
    return data, sorted(list(domains)), sorted(list(realms))

svc_data, domains, realms = load_svc_dataset('train_svc.jsonl')
DOMAIN_TO_IDX = {d: i for i, d in enumerate(domains)}
IDX_TO_DOMAIN = {i: d for i, d in enumerate(domains)}
REALM_TO_IDX = {r: i for i, r in enumerate(realms)}
IDX_TO_REALM = {i: r for i, r in enumerate(realms)}

with open('svc_label_maps.json', 'w', encoding='utf-8') as f:
    json.dump({"DOMAIN_TO_IDX": DOMAIN_TO_IDX, "IDX_TO_DOMAIN": IDX_TO_DOMAIN,
               "REALM_TO_IDX": REALM_TO_IDX, "IDX_TO_REALM": IDX_TO_REALM}, f)

# --- 2. SBERT & SVC Embeddings ---
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
y_realm = np.array([REALM_TO_IDX[rec['realm']] for rec in svc_data])
y_diff = np.array([rec['metadata']['difficulty'] for rec in svc_data])

X = torch.tensor(X, dtype=torch.float32)
y_domain = torch.tensor(y_domain, dtype=torch.long)
y_realm = torch.tensor(y_realm, dtype=torch.long)
y_diff = torch.tensor(y_diff, dtype=torch.float32)

input_dim = X.size(1)
num_domains = len(domains)
num_realms = len(realms)

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
y_realm_train = y_realm[train_idx]
y_diff_train = y_diff[train_idx]
X_test = X[test_idx]
y_domain_test = y_domain[test_idx]
y_realm_test = y_realm[test_idx]
y_diff_test = y_diff[test_idx]
test_data = [svc_data[i] for i in test_idx]

# --- 4. Classifier/Regressor Classes ---
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
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
clf_domain = Classifier(input_dim, num_domains)
opt_domain = torch.optim.Adam(clf_domain.parameters(), lr=0.01)
loss_domain = nn.CrossEntropyLoss()

for epoch in range(60):
    opt_domain.zero_grad()
    logits = clf_domain(X_train)
    loss = loss_domain(logits, y_domain_train)
    loss.backward()
    opt_domain.step()
    if epoch % 15 == 0 or epoch == 59:
        preds = logits.argmax(dim=1)
        acc = (preds == y_domain_train).float().mean().item()
        print(f"Epoch {epoch} | Train Domain Acc: {acc:.3f} | Loss: {loss.item():.3f}")

# --- 6. Train Realm Classifier ---
clf_realm = Classifier(input_dim, num_realms)
opt_realm = torch.optim.Adam(clf_realm.parameters(), lr=0.01)
loss_realm = nn.CrossEntropyLoss()

for epoch in range(60):
    opt_realm.zero_grad()
    logits = clf_realm(X_train)
    loss = loss_realm(logits, y_realm_train)
    loss.backward()
    opt_realm.step()
    if epoch % 15 == 0 or epoch == 59:
        preds = logits.argmax(dim=1)
        acc = (preds == y_realm_train).float().mean().item()
        print(f"Epoch {epoch} | Train Realm Acc: {acc:.3f} | Loss: {loss.item():.3f}")

# --- 7. Train Difficulty Regressor ---
regressor = DifficultyRegressor(input_dim)
opt_reg = torch.optim.Adam(regressor.parameters(), lr=0.01)
loss_reg = nn.MSELoss()
for epoch in range(60):
    opt_reg.zero_grad()
    output = regressor(X_train)
    loss = loss_reg(output, y_diff_train)
    loss.backward()
    opt_reg.step()
    if epoch % 15 == 0 or epoch == 59:
        print(f"Epoch {epoch} | Train Diff MSE Loss: {loss.item():.4f}")

print("Models and label mappings saved.")
torch.save(clf_domain.state_dict(), 'svc_domain_classifier.pt')
torch.save(clf_realm.state_dict(), 'svc_realm_classifier.pt')
torch.save(regressor.state_dict(), 'svc_difficulty_regressor.pt')

# --- 8. Reload & Evaluate/Print Accuracy and Loss Stats ---
def load_classifier(input_dim, num_classes, filename):
    clf = Classifier(input_dim, num_classes)
    clf.load_state_dict(torch.load(filename))
    clf.eval()
    return clf

def load_regressor(input_dim, filename):
    reg = DifficultyRegressor(input_dim)
    reg.load_state_dict(torch.load(filename))
    reg.eval()
    return reg

def load_label_maps(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        m = json.load(f)
    return {int(k): v for k, v in m['IDX_TO_DOMAIN'].items()}, {int(k): v for k, v in m['IDX_TO_REALM'].items()}

IDX_TO_DOMAIN, IDX_TO_REALM = load_label_maps('svc_label_maps.json')

clf_domain2 = load_classifier(input_dim, num_domains, 'svc_domain_classifier.pt')
clf_realm2 = load_classifier(input_dim, num_realms, 'svc_realm_classifier.pt')
regressor2 = load_regressor(input_dim, 'svc_difficulty_regressor.pt')

with torch.no_grad():
    # Domain/Realm accuracy (test and train)
    logits_domain_test = clf_domain2(X_test)
    preds_domain_test = logits_domain_test.argmax(dim=1)
    domain_acc_test = (preds_domain_test == y_domain_test).float().mean().item()
    domain_acc_train = (clf_domain2(X_train).argmax(dim=1) == y_domain_train).float().mean().item()

    logits_realm_test = clf_realm2(X_test)
    preds_realm_test = logits_realm_test.argmax(dim=1)
    realm_acc_test = (preds_realm_test == y_realm_test).float().mean().item()
    realm_acc_train = (clf_realm2(X_train).argmax(dim=1) == y_realm_train).float().mean().item()

    # Difficulty MSE stats (train/test)
    output_diff_test = regressor2(X_test)
    diff_mse_test = nn.functional.mse_loss(output_diff_test, y_diff_test).item()
    output_diff_train = regressor2(X_train)
    diff_mse_train = nn.functional.mse_loss(output_diff_train, y_diff_train).item()



for i, rec in enumerate(test_data):
    svc = rec['metadata']['svc']
    print("-" * 40)
    print(f"Text: {rec['text']}")
    print(f"SVC: Subject: {svc['subject']} | Verb: {svc['verb']} | Complement: {svc['complement']}")
    print(f"True domain: {rec['metadata']['domain']} | Pred domain: {IDX_TO_DOMAIN.get(preds_domain_test[i].item(), 'UNKNOWN')}")
    print(f"True realm: {rec['realm']} | Pred realm: {IDX_TO_REALM.get(preds_realm_test[i].item(), 'UNKNOWN')}")
    print(f"True difficulty: {rec['metadata']['difficulty']:.2f} | Pred difficulty: {float(output_diff_test[i]):.2f}\n")

print(f"\nDomain Acc | Train: {domain_acc_train:.3f} | Test: {domain_acc_test:.3f}")
print(f"Realm Acc  | Train: {realm_acc_train:.3f} | Test: {realm_acc_test:.3f}")
print(f"Difficulty MSE Loss | Train: {diff_mse_train:.4f} | Test: {diff_mse_test:.4f}\n")