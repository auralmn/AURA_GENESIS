#!/usr/bin/env python3
"""
MoE Calibration Helper
Auto-sweeps (temperature, usage_beta, top_k) to hit target entropy/usage
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Add project root to Python path
sys.path.insert(0, '/Volumes/Others2/AURA_GENESIS')

def load_instruct_samples(instruct_file: str, max_samples: int = 2000) -> List[str]:
    """Load instruction samples from JSONL file"""
    samples = []
    with open(instruct_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                if 'instruction' in data:
                    samples.append(data['instruction'])
                elif 'text' in data:
                    samples.append(data['text'])
            except Exception:
                continue
    return samples

def evaluate_moe_config(router, samples: List[str], config: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate MoE configuration on samples"""
    # Update router configuration
    if 'temperature' in config:
        router.gating.temperature = config['temperature']
    if 'usage_beta' in config:
        router.gating.usage_beta = config['usage_beta']
    if 'top_k' in config:
        router.gating.top_k = config['top_k']
        router.top_k = config['top_k']
    
    # Reset router state
    router.reset()
    
    # Process samples
    usage_counts = np.zeros(len(router.names))
    total_energy = 0.0
    routing_confidences = []
    
    for sample in samples:
        try:
            # Get features (simplified - you might need to adapt this)
            feats = np.random.randn(384)  # Placeholder - replace with actual feature extraction
            
            # Route through MoE
            result = router.route(feats, attn_gain=1.0)
            
            # Track usage
            for name, info in result['per_expert'].items():
                if info['gate'] > 0:
                    idx = router.names.index(name)
                    usage_counts[idx] += 1
            
            # Track energy
            total_energy += result.get('energy_j', 0.0)
            
            # Track routing confidence (simplified)
            routing_confidences.append(float(np.max(result['probs'])))
            
        except Exception as e:
            print(f"Warning: Error processing sample: {e}")
            continue
    
    # Calculate metrics
    usage_probs = usage_counts / max(1, usage_counts.sum())
    target_usage = 1.0 / len(router.names)
    
    metrics = {
        'usage_std': float(np.std(usage_probs)),
        'usage_entropy': float(-np.sum(usage_probs * np.log(usage_probs + 1e-12))),
        'target_usage': target_usage,
        'usage_imbalance': float(np.max(usage_probs) - target_usage),
        'avg_confidence': float(np.mean(routing_confidences)) if routing_confidences else 0.0,
        'energy_per_query': total_energy / max(1, len(samples)),
        'total_queries': len(samples)
    }
    
    return metrics

def grid_search_moe_configs(router, samples: List[str], 
                          temp_range: Tuple[float, float, int] = (1.0, 1.0, 1),
                          beta_range: Tuple[float, float, int] = (0.5, 0.5, 1),
                          top_k_range: Tuple[int, int] = (2, 2)) -> List[Dict[str, Any]]:
    """Grid search over MoE hyperparameters"""
    
    temp_values = np.linspace(temp_range[0], temp_range[1], temp_range[2])
    beta_values = np.linspace(beta_range[0], beta_range[1], beta_range[2])
    top_k_values = list(range(top_k_range[0], top_k_range[1] + 1))
    
    results = []
    total_configs = len(temp_values) * len(beta_values) * len(top_k_values)
    
    print(f"🔍 Grid searching {total_configs} configurations...")
    
    for i, temp in enumerate(temp_values):
        for j, beta in enumerate(beta_values):
            for k, top_k in enumerate(top_k_values):
                config = {
                    'temperature': float(temp),
                    'usage_beta': float(beta),
                    'top_k': int(top_k)
                }
                
                print(f"  Testing config {i*len(beta_values)*len(top_k_values) + j*len(top_k_values) + k + 1}/{total_configs}: "
                      f"T={temp:.2f}, β={beta:.2f}, K={top_k}")
                
                metrics = evaluate_moe_config(router, samples, config)
                
                result = {
                    'config': config,
                    'metrics': metrics,
                    'score': calculate_config_score(metrics)
                }
                results.append(result)
    
    return results

def calculate_config_score(metrics: Dict[str, float]) -> float:
    """Calculate overall score for configuration"""
    # Weighted combination of metrics
    usage_score = 1.0 - metrics['usage_imbalance']  # Lower imbalance is better
    entropy_score = metrics['usage_entropy'] / np.log(len(metrics.get('target_usage', 6)))  # Normalized entropy
    confidence_score = metrics['avg_confidence']
    
    # Energy penalty (lower is better)
    energy_penalty = min(1.0, 1.0 / (1.0 + metrics['energy_per_query'] * 1000))  # Scale energy
    
    # Combined score
    score = 0.4 * usage_score + 0.3 * entropy_score + 0.2 * confidence_score + 0.1 * energy_penalty
    return float(score)

def find_best_configs(results: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """Find best configurations by score"""
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    return sorted_results[:top_n]

def print_config_report(best_configs: List[Dict[str, Any]]):
    """Print detailed report of best configurations"""
    print(f"\n🏆 Top {len(best_configs)} MoE Configurations:")
    print("=" * 80)
    
    for i, result in enumerate(best_configs):
        config = result['config']
        metrics = result['metrics']
        score = result['score']
        
        print(f"\n#{i+1} Score: {score:.4f}")
        print(f"  Config: T={config['temperature']:.2f}, β={config['usage_beta']:.2f}, K={config['top_k']}")
        print(f"  Usage std: {metrics['usage_std']:.4f} (target: {metrics['target_usage']:.4f})")
        print(f"  Usage entropy: {metrics['usage_entropy']:.4f}")
        print(f"  Usage imbalance: {metrics['usage_imbalance']:.4f}")
        print(f"  Avg confidence: {metrics['avg_confidence']:.4f}")
        print(f"  Energy/query: {metrics['energy_per_query']:.6f} J")

def save_calibration_results(best_configs: List[Dict[str, Any]], output_file: str):
    """Save calibration results to JSON file"""
    results_data = {
        'calibration_timestamp': str(np.datetime64('now')),
        'best_configs': best_configs,
        'recommendations': {
            'primary_config': best_configs[0]['config'] if best_configs else {},
            'usage_beta_range': [0.3, 0.8],
            'temperature_range': [1.0, 1.3],
            'top_k_recommendations': [2, 3]
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Calibration results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="MoE Calibration Helper")
    parser.add_argument('--instruct-file', required=True, 
                       help='Path to instruct dataset JSONL file')
    parser.add_argument('--max-samples', type=int, default=2000,
                       help='Maximum samples to use for calibration')
    parser.add_argument('--output', default='moe_calibration_results.json',
                       help='Output file for calibration results')
    parser.add_argument('--temp-range', nargs=3, type=float, default=[1.0, 1.0, 1],
                       help='Temperature range: min max steps (AURA uses fixed values)')
    parser.add_argument('--beta-range', nargs=3, type=float, default=[0.5, 0.5, 1],
                       help='Usage beta range: min max steps (AURA uses fixed values)')
    parser.add_argument('--top-k-range', nargs=2, type=int, default=[2, 2],
                       help='Top-K range: min max (AURA uses fixed values)')
    parser.add_argument('--top-n', type=int, default=5,
                       help='Number of best configs to report')
    
    args = parser.parse_args()
    
    print("🔧 MoE Calibration Helper")
    print("=" * 50)
    
    # Load samples
    print(f"📚 Loading samples from {args.instruct_file}...")
    samples = load_instruct_samples(args.instruct_file, args.max_samples)
    print(f"   Loaded {len(samples)} samples")
    
    if len(samples) < 10:
        print("❌ Not enough samples for calibration. Need at least 10.")
        return 1
    
    # Create mock router for testing
    # Note: In production, you'd load your actual router here
    print("⚠️  Using mock router for calibration. In production, load your actual router.")
    
    # For now, create a simple mock router
    class MockRouter:
        def __init__(self):
            self.names = ['conversations', 'empathy', 'historical', 'inventions', 'movie_annotated', 'socratic']
            self.gating = type('MockGating', (), {
                'temperature': 1.0,
                'usage_beta': 0.5,
                'top_k': 2
            })()
            self.top_k = 2
        
        def reset(self):
            pass
        
        def route(self, feats, attn_gain=1.0):
            # Mock routing result
            n_experts = len(self.names)
            probs = np.random.dirichlet(np.ones(n_experts))
            top_k = min(self.top_k, n_experts)
            topk_idx = np.argpartition(probs, -top_k)[-top_k:]
            
            per_expert = {}
            for i in topk_idx:
                name = self.names[i]
                gate = probs[i] / probs[topk_idx].sum()
                per_expert[name] = {'gate': gate, 'pred': np.random.randn()}
            
            return {
                'per_expert': per_expert,
                'probs': probs,
                'energy_j': np.random.exponential(0.001)
            }
    
    router = MockRouter()
    
    # Run grid search
    results = grid_search_moe_configs(
        router, samples,
        temp_range=tuple(args.temp_range),
        beta_range=tuple(args.beta_range),
        top_k_range=tuple(args.top_k_range)
    )
    
    # Find best configurations
    best_configs = find_best_configs(results, args.top_n)
    
    # Print report
    print_config_report(best_configs)
    
    # Save results
    save_calibration_results(best_configs, args.output)
    
    print(f"\n✅ Calibration complete! Best config:")
    if best_configs:
        best = best_configs[0]['config']
        print(f"   Temperature: {best['temperature']:.2f}")
        print(f"   Usage Beta: {best['usage_beta']:.2f}")
        print(f"   Top-K: {best['top_k']}")
        print(f"\n🚀 Apply these settings to your router:")
        print(f"   router.gating.temperature = {best['temperature']:.2f}")
        print(f"   router.gating.usage_beta = {best['usage_beta']:.2f}")
        print(f"   router.gating.top_k = {best['top_k']}")
        print(f"   router.top_k = {best['top_k']}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
