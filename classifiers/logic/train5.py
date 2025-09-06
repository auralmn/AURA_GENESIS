# Complete Integration Example: Enhanced SVC Pipeline with Linguistic Features
# This is your modified pipeline that uses the enhanced dataset

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer

# ==========================================
# ENHANCED DATA LOADING FUNCTIONS
# ==========================================

def load_enhanced_svc_dataset(filename):
    """Load the enhanced SVC dataset with linguistic features"""
    data = []
    domains = set()
    realms = set()
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            data.append(record)
            domains.add(record['metadata']['domain'])
            realms.add(record['realm'])
    
    print(f"Loaded {len(data)} enhanced records")
    print("Available features in enhanced data:")
    if data:
        sample = data[0]
        print(f"  - linguistic_features: {list(sample['linguistic_features'].keys())}")
        print(f"  - svc_linguistics: {list(sample['svc_linguistics'].keys())}")
        print(f"  - tagged_versions: {list(sample['tagged_versions'].keys())}")
        print(f"  - structural_features: {list(sample['structural_features'].keys())}")
    
    return data, sorted(list(domains)), sorted(list(realms))


# ==========================================
# ENHANCED FEATURE EXTRACTION FUNCTIONS
# ==========================================

def extract_pos_features(pos_tags):
    """Extract POS tag distribution features"""
    pos_types = ['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET', 'PRON', 'NUM']
    pos_counts = np.zeros(len(pos_types), dtype=np.float32)
    
    for pos_info in pos_tags:
        pos = pos_info['pos']
        if pos in pos_types:
            idx = pos_types.index(pos)
            pos_counts[idx] += 1
    
    # Normalize by total tokens
    total = sum(pos_counts)
    if total > 0:
        pos_counts = pos_counts / total
    
    return pos_counts


def extract_ner_features(named_entities):
    """Extract named entity distribution features"""
    entity_types = ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'MISC']
    ner_counts = np.zeros(len(entity_types), dtype=np.float32)
    
    for ent_info in named_entities:
        ent_type = ent_info['type']
        if ent_type in entity_types:
            idx = entity_types.index(ent_type)
            ner_counts[idx] += 1
    
    return ner_counts


def extract_structural_features(svc_linguistics):
    """Extract SVC structural features"""
    features = []
    
    # Subject complexity
    subj_analysis = svc_linguistics['subject_analysis']
    features.append(subj_analysis['complexity_score'])
    features.append(len(subj_analysis['tokens']))
    
    # Verb complexity  
    verb_analysis = svc_linguistics['verb_analysis']
    features.append(verb_analysis['complexity_score'])
    features.append(len(verb_analysis['tokens']))
    
    # Complement complexity
    comp_analysis = svc_linguistics['complement_analysis']
    features.append(comp_analysis['complexity_score'])
    features.append(len(comp_analysis['tokens']))
    
    return np.array(features, dtype=np.float32)


def extract_morphological_features(structural_features):
    """Extract morphological complexity features"""
    features = []
    
    # Syntactic complexity
    syntax_complex = structural_features['syntactic_complexity']
    features.append(syntax_complex['avg_dependency_depth'])
    features.append(syntax_complex['clause_count'])
    features.append(syntax_complex['subordination_ratio'])
    
    # SVC balance
    svc_balance = structural_features['svc_balance']
    features.append(svc_balance['balance_score'])
    features.append(svc_balance['subject_ratio'])
    features.append(svc_balance['complement_ratio'])
    
    # Verb features count
    verb_features = structural_features['verb_tense_info']
    features.append(len(verb_features))
    
    return np.array(features, dtype=np.float32)


def extract_linguistic_context_features(record):
    """Extract high-level linguistic context features"""
    features = []
    
    ling_features = record['linguistic_features']
    
    # Text statistics
    features.append(ling_features['sentence_count'])
    features.append(ling_features['word_count'])
    features.append(ling_features['word_count'] / max(ling_features['sentence_count'], 1))
    
    # Linguistic diversity
    pos_diversity = len(set(tag['pos'] for tag in ling_features['pos_tags']))
    features.append(pos_diversity)
    
    # Entity density
    entity_density = len(ling_features['named_entities']) / max(ling_features['word_count'], 1)
    features.append(entity_density)
    
    return np.array(features, dtype=np.float32)


# ==========================================
# ENHANCED EMBEDDING FUNCTIONS
# ==========================================

def get_enhanced_svc_embedding(record, sbert_model):
    """Create embeddings that include linguistic features"""
    
    # Original SVC embeddings (your existing code)
    svc = record['metadata']['svc']
    subj_emb = sbert_model.encode(svc['subject'])
    verb_emb = sbert_model.encode(svc['verb'])
    comp_emb = sbert_model.encode(svc['complement'])
    
    # NEW: Add linguistic feature embeddings
    linguistic_features = record['linguistic_features']
    
    # POS tag features (count of each POS type)
    pos_counts = extract_pos_features(linguistic_features['pos_tags'])
    
    # NER features (count of each entity type)  
    ner_counts = extract_ner_features(linguistic_features['named_entities'])
    
    # Structural features from SVC analysis
    structural_features = extract_structural_features(record['svc_linguistics'])
    
    # Tagged text embeddings (use structural tags)
    tagged_text = record['tagged_versions']['svc_full_tagged']
    tagged_emb = sbert_model.encode(tagged_text)
    
    # Morphological complexity features
    morph_features = extract_morphological_features(record['structural_features'])
    
    # Combine all features
    combined_embedding = np.concatenate([
        subj_emb,           # Original subject embedding
        verb_emb,           # Original verb embedding  
        comp_emb,           # Original complement embedding
        tagged_emb,         # NEW: Tagged structure embedding
        pos_counts,         # NEW: POS distribution features
        ner_counts,         # NEW: NER distribution features
        structural_features, # NEW: SVC structural features
        morph_features      # NEW: Morphological features
    ])
    
    return combined_embedding


def get_enhanced_full_knowledge_embedding(record, sbert_model):
    """Your existing function enhanced with linguistic features"""
    
    # Original text embedding
    text_emb = sbert_model.encode(record['text'])
    
    # Enhanced SVC embedding (includes linguistic features)
    svc_emb = get_enhanced_svc_embedding(record, sbert_model)
    
    # Original metadata features
    domain = record['metadata']['domain']
    diff = record['metadata']['difficulty']
    domain_onehot = np.zeros(len(DOMAIN_TO_IDX), dtype=np.float32)
    domain_onehot[DOMAIN_TO_IDX[domain]] = 1
    
    # NEW: Additional linguistic context features
    linguistic_context = extract_linguistic_context_features(record)
    
    return np.concatenate([
        text_emb,               # Original text embedding
        svc_emb,               # Enhanced SVC embedding with linguistics
        domain_onehot,         # Original domain one-hot
        [diff],                # Original difficulty
        linguistic_context     # NEW: Linguistic context features
    ])


# ==========================================
# YOUR EXISTING MODEL CLASSES (UNCHANGED)
# ==========================================

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


# ==========================================
# ENHANCED MAIN PIPELINE
# ==========================================

def main():
    """Your complete pipeline enhanced with linguistic features"""
    
    # --- 1. Load Enhanced SVC dataset ---
    # CHANGE: Use enhanced dataset loader
    svc_data, domains, realms = load_enhanced_svc_dataset('train_svc_enhanced.jsonl')
    
    # Create label mappings (same as before)
    global DOMAIN_TO_IDX, IDX_TO_DOMAIN, REALM_TO_IDX, IDX_TO_REALM
    DOMAIN_TO_IDX = {d: i for i, d in enumerate(domains)}
    IDX_TO_DOMAIN = {i: d for i, d in enumerate(domains)}
    REALM_TO_IDX = {r: i for i, r in enumerate(realms)}
    IDX_TO_REALM = {i: r for i, r in enumerate(realms)}

    with open('svc_label_maps.json', 'w', encoding='utf-8') as f:
        json.dump({"DOMAIN_TO_IDX": DOMAIN_TO_IDX, "IDX_TO_DOMAIN": IDX_TO_DOMAIN,
                   "REALM_TO_IDX": REALM_TO_IDX, "IDX_TO_REALM": IDX_TO_REALM}, f)

    # --- 2. SBERT & Enhanced SVC Embeddings ---
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating enhanced embeddings with linguistic features...")
    X = np.array([get_enhanced_full_knowledge_embedding(rec, sbert) for rec in svc_data])
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

    print(f"Enhanced dataset prepared:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {input_dim} (vs ~1150 in original)")
    print(f"  Additional linguistic features: {input_dim - 1150}")
    print(f"  Domains: {num_domains}, Realms: {num_realms}")

    # --- 3. Dataset Split (unchanged) ---
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

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # --- 4. Train Domain Classifier (unchanged architecture) ---
    clf_domain = Classifier(input_dim, num_domains)
    opt_domain = torch.optim.Adam(clf_domain.parameters(), lr=0.01)
    loss_domain = nn.CrossEntropyLoss()

    print("\nTraining Domain Classifier...")
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

    # --- 5. Train Realm Classifier (unchanged architecture) ---
    clf_realm = Classifier(input_dim, num_realms)
    opt_realm = torch.optim.Adam(clf_realm.parameters(), lr=0.01)
    loss_realm = nn.CrossEntropyLoss()

    print("\nTraining Realm Classifier...")
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

    # --- 6. Train Difficulty Regressor (unchanged architecture) ---
    regressor = DifficultyRegressor(input_dim)
    opt_reg = torch.optim.Adam(regressor.parameters(), lr=0.01)
    loss_reg = nn.MSELoss()
    
    print("\nTraining Difficulty Regressor...")
    for epoch in range(60):
        opt_reg.zero_grad()
        output = regressor(X_train)
        loss = loss_reg(output, y_diff_train)
        loss.backward()
        opt_reg.step()
        if epoch % 15 == 0 or epoch == 59:
            print(f"Epoch {epoch} | Train Diff MSE Loss: {loss.item():.4f}")

    # --- 7. Save Models ---
    print("Saving enhanced models...")
    torch.save(clf_domain.state_dict(), 'svc_domain_classifier_enhanced.pt')
    torch.save(clf_realm.state_dict(), 'svc_realm_classifier_enhanced.pt')
    torch.save(regressor.state_dict(), 'svc_difficulty_regressor_enhanced.pt')

    # --- 8. Evaluation with Enhanced Features ---
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

    clf_domain2 = load_classifier(input_dim, num_domains, 'svc_domain_classifier_enhanced.pt')
    clf_realm2 = load_classifier(input_dim, num_realms, 'svc_realm_classifier_enhanced.pt')
    regressor2 = load_regressor(input_dim, 'svc_difficulty_regressor_enhanced.pt')

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

    # --- 9. Enhanced Results Display ---
    print("\n" + "="*60)
    print("ENHANCED SVC PIPELINE RESULTS")
    print("="*60)
    
    for i, rec in enumerate(test_data[:5]):  # Show first 5 examples
        svc = rec['metadata']['svc']
        print(f"\n--- Example {i+1} ---")
        print(f"Text: {rec['text']}")
        print(f"SVC: Subject: {svc['subject']} | Verb: {svc['verb']} | Complement: {svc['complement']}")
        
        # Show linguistic features
        ling_features = rec['linguistic_features']
        print(f"Linguistic: {ling_features['word_count']} words, {ling_features['sentence_count']} sentences")
        print(f"POS diversity: {len(set(tag['pos'] for tag in ling_features['pos_tags']))} types")
        print(f"Named entities: {len(ling_features['named_entities'])}")
        
        # Show tagged version
        tagged_version = rec['tagged_versions']['svc_full_tagged']
        print(f"Tagged SVC: {tagged_version}")
        
        # Show predictions
        print(f"True domain: {rec['metadata']['domain']} | Pred domain: {IDX_TO_DOMAIN.get(preds_domain_test[i].item(), 'UNKNOWN')}")
        print(f"True realm: {rec['realm']} | Pred realm: {IDX_TO_REALM.get(preds_realm_test[i].item(), 'UNKNOWN')}")
        print(f"True difficulty: {rec['metadata']['difficulty']:.2f} | Pred difficulty: {float(output_diff_test[i]):.2f}")

    print(f"\n" + "="*60)
    print("FINAL PERFORMANCE METRICS (ENHANCED)")
    print("="*60)
    print(f"Domain Acc | Train: {domain_acc_train:.3f} | Test: {domain_acc_test:.3f}")
    print(f"Realm Acc  | Train: {realm_acc_train:.3f} | Test: {realm_acc_test:.3f}")
    print(f"Difficulty MSE Loss | Train: {diff_mse_train:.4f} | Test: {diff_mse_test:.4f}")
    
    print(f"\nFeature breakdown:")
    print(f"  Total features: {input_dim}")
    print(f"  Text embeddings: 384")
    print(f"  SVC embeddings: 1152 (384*3)")
    print(f"  Tagged embeddings: 384")
    print(f"  POS features: 8")
    print(f"  NER features: 6")
    print(f"  Structural features: 6")
    print(f"  Morphological features: 7")
    print(f"  Context features: 5")
    print(f"  Domain one-hot: {len(domains)}")
    print(f"  Difficulty: 1")
    
    print(f"\n🚀 ENHANCED SVC PIPELINE COMPLETE! 🚀")


if __name__ == "__main__":
    # Prerequisites:
    # 1. pip install stanza torch sentence-transformers numpy
    # 2. Run: python svc-data-enhancer.py train_svc.jsonl train_svc_enhanced.jsonl
    # 3. Run this enhanced pipeline
    
    main()