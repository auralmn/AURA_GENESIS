# SVC Data Enhancement with Stanza Linguistic Annotations
# This script adds comprehensive linguistic features to your existing SVC training data

import json
import stanza
import copy
from typing import List, Dict, Any, Optional
import argparse
import os
from tqdm import tqdm

class StanzaSVCEnhancer:
    """
    Enhances SVC training data with comprehensive Stanza linguistic annotations
    """
    
    def __init__(self, language='en', verbose=False):
        """
        Initialize the Stanza pipeline
        
        Args:
            language: Language code (default: 'en')
            verbose: Enable verbose logging
        """
        self.language = language
        self.verbose = verbose
        
        print("Initializing Stanza pipeline...")
        print("Downloading language models if needed...")
        
        # Download and initialize Stanza
        stanza.download(language, verbose=verbose)
        self.nlp = stanza.Pipeline(
            language, 
            processors='tokenize,pos,lemma,ner,depparse',
            verbose=verbose,
            use_gpu=True  # Set to False if no GPU available
        )
        
        print("Stanza pipeline ready!")
    
    def enhance_svc_dataset(self, input_file: str, output_file: str, 
                           batch_size: int = 32) -> Dict[str, Any]:
        """
        Process entire SVC dataset and add linguistic annotations
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output enhanced JSONL file
            batch_size: Number of records to process at once
            
        Returns:
            Processing statistics
        """
        print(f"Processing {input_file} → {output_file}")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Load original data
        original_records = self._load_jsonl(input_file)
        print(f"Loaded {len(original_records)} records")
        
        # Process records in batches
        enhanced_records = []
        
        for i in tqdm(range(0, len(original_records), batch_size), 
                     desc="Processing batches"):
            batch = original_records[i:i+batch_size]
            enhanced_batch = self._process_batch(batch)
            enhanced_records.extend(enhanced_batch)
        
        # Save enhanced data
        self._save_jsonl(enhanced_records, output_file)
        
        # Generate statistics
        stats = self._generate_statistics(original_records, enhanced_records)
        
        print(f"Enhanced dataset saved to {output_file}")
        print(f"Processing complete: {len(enhanced_records)} records enhanced")
        
        return stats
    
    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """Load records from JSONL file"""
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        return records
    
    def _save_jsonl(self, records: List[Dict], file_path: str):
        """Save records to JSONL file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of records"""
        enhanced_batch = []
        
        for record in batch:
            try:
                enhanced_record = self._enhance_single_record(record)
                enhanced_batch.append(enhanced_record)
            except Exception as e:
                if self.verbose:
                    print(f"Error processing record: {e}")
                # Keep original record if enhancement fails
                enhanced_batch.append(record)
        
        return enhanced_batch
    
    def _enhance_single_record(self, record: Dict) -> Dict:
        """
        Add comprehensive linguistic annotations to a single record
        """
        enhanced_record = copy.deepcopy(record)
        
        # Analyze main text
        text_doc = self.nlp(record['text'])
        enhanced_record['linguistic_features'] = self._extract_text_features(text_doc, record['text'])
        
        # Analyze SVC components
        svc = record['metadata']['svc']
        enhanced_record['svc_linguistics'] = {
            'subject_analysis': self._analyze_svc_component(svc['subject'], 'SUBJ'),
            'verb_analysis': self._analyze_svc_component(svc['verb'], 'VERB'),
            'complement_analysis': self._analyze_svc_component(svc['complement'], 'COMP')
        }
        
        # Create tagged versions
        enhanced_record['tagged_versions'] = self._create_tagged_versions(record, text_doc)
        
        # Add structural features
        enhanced_record['structural_features'] = self._extract_structural_features(text_doc, svc)
        
        return enhanced_record
    
    def _extract_text_features(self, doc, original_text: str) -> Dict:
        """Extract comprehensive linguistic features from text"""
        features = {
            'tokens': [],
            'pos_tags': [],
            'lemmas': [],
            'named_entities': [],
            'dependencies': [],
            'sentence_count': len(doc.sentences),
            'word_count': sum(len(sent.words) for sent in doc.sentences),
            'morphological_features': {}
        }
        
        # Process each sentence
        for sent_idx, sentence in enumerate(doc.sentences):
            # Extract tokens and POS tags
            for word in sentence.words:
                features['tokens'].append({
                    'text': word.text,
                    'sentence_id': sent_idx,
                    'token_id': word.id
                })
                
                features['pos_tags'].append({
                    'token': word.text,
                    'pos': word.upos,
                    'xpos': word.xpos,
                    'feats': word.feats,
                    'sentence_id': sent_idx,
                    'token_id': word.id
                })
                
                features['lemmas'].append({
                    'token': word.text,
                    'lemma': word.lemma,
                    'sentence_id': sent_idx,
                    'token_id': word.id
                })
            
            # Extract dependencies
            for word in sentence.words:
                features['dependencies'].append({
                    'token': word.text,
                    'head': sentence.words[word.head-1].text if word.head > 0 else 'ROOT',
                    'deprel': word.deprel,
                    'sentence_id': sent_idx,
                    'token_id': word.id
                })
        
        # Extract named entities
        for sentence in doc.sentences:
            for ent in sentence.ents:
                features['named_entities'].append({
                    'text': ent.text,
                    'type': ent.type,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char
                })
        
        # Extract morphological features
        features['morphological_features'] = self._extract_morphological_features(doc)
        
        return features
    
    def _analyze_svc_component(self, component: str, svc_type: str) -> Dict:
        """Analyze individual SVC components with Stanza"""
        doc = self.nlp(component)
        
        analysis = {
            'component': component,
            'svc_type': svc_type,
            'structural_tag': f'[{svc_type}]',
            'tokens': [],
            'pos_tags': [],
            'lemmas': [],
            'dependencies': [],
            'complexity_score': 0
        }
        
        # Extract features for this component
        for sentence in doc.sentences:
            for word in sentence.words:
                analysis['tokens'].append(word.text)
                analysis['pos_tags'].append({
                    'token': word.text,
                    'pos': word.upos,
                    'feats': word.feats
                })
                analysis['lemmas'].append({
                    'token': word.text,
                    'lemma': word.lemma
                })
                analysis['dependencies'].append({
                    'token': word.text,
                    'head': sentence.words[word.head-1].text if word.head > 0 else 'ROOT',
                    'deprel': word.deprel
                })
        
        # Calculate complexity score
        analysis['complexity_score'] = self._calculate_complexity(doc)
        
        return analysis
    
    def _create_tagged_versions(self, record: Dict, text_doc) -> Dict:
        """Create multiple tagged versions of the text and SVC"""
        svc = record['metadata']['svc']
        
        tagged_versions = {
            # SVC structural tagging
            'svc_full_tagged': f"[SUBJ]{svc['subject']}[/SUBJ] [VERB]{svc['verb']}[/VERB] [COMP]{svc['complement']}[/COMP]",
            'svc_simple_tagged': f"{svc['subject']} [VERB] {svc['complement']}",
            'svc_pattern': "[SUBJ]-[VERB]-[COMP]",
            
            # Text POS tagging
            'pos_tagged_text': self._create_pos_tagged_text(text_doc),
            'lemma_tagged_text': self._create_lemma_tagged_text(text_doc),
            'ner_tagged_text': self._create_ner_tagged_text(text_doc),
            
            # Dependency structure
            'dependency_pattern': self._extract_dependency_pattern(text_doc),
            
            # Semantic roles
            'semantic_roles': {
                'agent': svc['subject'],
                'action': svc['verb'], 
                'theme': svc['complement'],
                'domain': record['metadata']['domain'],
                'realm': record['realm']
            },
            
            # Combined structural representation
            'structural_representation': self._create_structural_representation(svc, text_doc)
        }
        
        return tagged_versions
    
    def _create_pos_tagged_text(self, doc) -> str:
        """Create POS-tagged version of text"""
        tagged_tokens = []
        for sentence in doc.sentences:
            sent_tokens = []
            for word in sentence.words:
                sent_tokens.append(f"{word.text}/{word.upos}")
            tagged_tokens.append(" ".join(sent_tokens))
        return " . ".join(tagged_tokens)
    
    def _create_lemma_tagged_text(self, doc) -> str:
        """Create lemmatized version of text"""
        lemma_tokens = []
        for sentence in doc.sentences:
            sent_tokens = []
            for word in sentence.words:
                sent_tokens.append(f"{word.text}→{word.lemma}")
            lemma_tokens.append(" ".join(sent_tokens))
        return " . ".join(lemma_tokens)
    
    def _create_ner_tagged_text(self, doc) -> str:
        """Create NER-tagged version of text"""
        ner_text = ""
        for sentence in doc.sentences:
            sent_text = sentence.text
            for ent in sentence.ents:
                sent_text = sent_text.replace(ent.text, f"[{ent.type}]{ent.text}[/{ent.type}]")
            ner_text += sent_text + " "
        return ner_text.strip()
    
    def _extract_dependency_pattern(self, doc) -> List[str]:
        """Extract dependency patterns"""
        patterns = []
        for sentence in doc.sentences:
            sent_pattern = []
            for word in sentence.words:
                sent_pattern.append(f"{word.upos}:{word.deprel}")
            patterns.append(" ".join(sent_pattern))
        return patterns
    
    def _create_structural_representation(self, svc: Dict, doc) -> str:
        """Create comprehensive structural representation"""
        # Extract main verb and its arguments
        main_verbs = []
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos == 'VERB' and 'root' in word.deprel:
                    main_verbs.append(word.text)
        
        structure = {
            'svc_pattern': f"[SUBJ:{len(svc['subject'].split())}] [VERB:{len(svc['verb'].split())}] [COMP:{len(svc['complement'].split())}]",
            'main_verbs': main_verbs,
            'sentence_structure': f"[{doc.sentences[0].words[0].upos}]-[{doc.sentences[0].words[-1].upos}]" if doc.sentences else ""
        }
        
        return json.dumps(structure)
    
    def _extract_structural_features(self, doc, svc: Dict) -> Dict:
        """Extract structural linguistic features"""
        return {
            'verb_tense_info': self._extract_verb_features(doc),
            'noun_phrase_structure': self._extract_np_structure(doc),
            'syntactic_complexity': self._calculate_syntactic_complexity(doc),
            'svc_balance': self._calculate_svc_balance(svc),
            'discourse_markers': self._extract_discourse_markers(doc)
        }
    
    def _extract_morphological_features(self, doc) -> Dict:
        """Extract morphological features"""
        features = {
            'verb_forms': [],
            'noun_forms': [],
            'adjective_forms': [],
            'complexity_metrics': {}
        }
        
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos == 'VERB':
                    features['verb_forms'].append({
                        'text': word.text,
                        'lemma': word.lemma,
                        'feats': word.feats
                    })
                elif word.upos == 'NOUN':
                    features['noun_forms'].append({
                        'text': word.text,
                        'lemma': word.lemma,
                        'feats': word.feats
                    })
                elif word.upos == 'ADJ':
                    features['adjective_forms'].append({
                        'text': word.text,
                        'lemma': word.lemma,
                        'feats': word.feats
                    })
        
        return features
    
    def _extract_verb_features(self, doc) -> List[Dict]:
        """Extract verb-specific features"""
        verb_features = []
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos == 'VERB':
                    verb_features.append({
                        'text': word.text,
                        'lemma': word.lemma,
                        'tense': self._extract_tense(word.feats),
                        'voice': self._extract_voice(word.feats),
                        'mood': self._extract_mood(word.feats)
                    })
        return verb_features
    
    def _extract_np_structure(self, doc) -> List[str]:
        """Extract noun phrase structures"""
        np_structures = []
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos == 'NOUN' and word.deprel in ['nsubj', 'obj', 'nmod']:
                    # Find dependents to build NP structure
                    dependents = [w for w in sentence.words if w.head == word.id]
                    structure = f"{word.upos}({len(dependents)})"
                    np_structures.append(structure)
        return np_structures
    
    def _calculate_complexity(self, doc) -> float:
        """Calculate linguistic complexity score"""
        if not doc.sentences:
            return 0.0
        
        total_words = sum(len(sent.words) for sent in doc.sentences)
        total_deps = sum(len([w for w in sent.words if w.deprel not in ['punct']]) for sent in doc.sentences)
        
        return total_deps / total_words if total_words > 0 else 0.0
    
    def _calculate_syntactic_complexity(self, doc) -> Dict:
        """Calculate syntactic complexity metrics"""
        metrics = {
            'avg_dependency_depth': 0,
            'clause_count': 0,
            'subordination_ratio': 0
        }
        
        if not doc.sentences:
            return metrics
        
        depths = []
        clauses = 0
        subordinations = 0
        
        for sentence in doc.sentences:
            for word in sentence.words:
                # Calculate dependency depth
                depth = self._get_dependency_depth(word, sentence.words)
                depths.append(depth)
                
                # Count clauses and subordinations
                if word.deprel in ['ccomp', 'xcomp', 'advcl']:
                    subordinations += 1
                if word.upos == 'VERB':
                    clauses += 1
        
        metrics['avg_dependency_depth'] = sum(depths) / len(depths) if depths else 0
        metrics['clause_count'] = clauses
        metrics['subordination_ratio'] = subordinations / clauses if clauses > 0 else 0
        
        return metrics
    
    def _calculate_svc_balance(self, svc: Dict) -> Dict:
        """Calculate balance metrics for SVC components"""
        subj_len = len(svc['subject'].split())
        verb_len = len(svc['verb'].split())
        comp_len = len(svc['complement'].split())
        
        total_len = subj_len + verb_len + comp_len
        
        return {
            'subject_ratio': subj_len / total_len,
            'verb_ratio': verb_len / total_len,
            'complement_ratio': comp_len / total_len,
            'balance_score': 1 - abs(subj_len - comp_len) / total_len
        }
    
    def _extract_discourse_markers(self, doc) -> List[str]:
        """Extract discourse markers and connectives"""
        markers = []
        discourse_words = ['however', 'therefore', 'moreover', 'furthermore', 'although', 'because']
        
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.text.lower() in discourse_words:
                    markers.append(word.text.lower())
        
        return markers
    
    def _extract_tense(self, feats: str) -> Optional[str]:
        """Extract tense from morphological features"""
        if not feats:
            return None
        if 'Tense=Past' in feats:
            return 'Past'
        elif 'Tense=Pres' in feats:
            return 'Present'
        elif 'Tense=Fut' in feats:
            return 'Future'
        return None
    
    def _extract_voice(self, feats: str) -> Optional[str]:
        """Extract voice from morphological features"""
        if not feats:
            return None
        if 'Voice=Pass' in feats:
            return 'Passive'
        elif 'Voice=Act' in feats:
            return 'Active'
        return None
    
    def _extract_mood(self, feats: str) -> Optional[str]:
        """Extract mood from morphological features"""
        if not feats:
            return None
        if 'Mood=Ind' in feats:
            return 'Indicative'
        elif 'Mood=Imp' in feats:
            return 'Imperative'
        elif 'Mood=Sub' in feats:
            return 'Subjunctive'
        return None
    
    def _get_dependency_depth(self, word, words) -> int:
        """Calculate dependency depth for a word"""
        depth = 0
        current = word
        
        while current.head > 0:
            depth += 1
            current = words[current.head - 1]
            if depth > 20:  # Prevent infinite loops
                break
        
        return depth
    
    def _generate_statistics(self, original: List[Dict], enhanced: List[Dict]) -> Dict:
        """Generate processing statistics"""
        stats = {
            'original_count': len(original),
            'enhanced_count': len(enhanced),
            'success_rate': len(enhanced) / len(original) if original else 0,
            'average_features_added': 0,
            'file_size_increase': 0
        }
        
        if enhanced:
            # Calculate average new features added
            original_keys = set(original[0].keys()) if original else set()
            enhanced_keys = set(enhanced[0].keys())
            new_keys = enhanced_keys - original_keys
            stats['average_features_added'] = len(new_keys)
            stats['new_feature_types'] = list(new_keys)
        
        return stats


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Enhance SVC training data with Stanza linguistic features')
    parser.add_argument('input_file', help='Input JSONL file path')
    parser.add_argument('output_file', help='Output enhanced JSONL file path')
    parser.add_argument('--language', default='en', help='Language code (default: en)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Initialize enhancer
    enhancer = StanzaSVCEnhancer(language=args.language, verbose=args.verbose)
    
    # Process dataset
    stats = enhancer.enhance_svc_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size
    )
    
    print("\nProcessing Statistics:")
    print(f"  Original records: {stats['original_count']}")
    print(f"  Enhanced records: {stats['enhanced_count']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  New features added: {stats['average_features_added']}")
    print(f"  New feature types: {stats.get('new_feature_types', [])}")


if __name__ == "__main__":
    main()