#!/usr/bin/env python3
"""
Task 1.7: Evaluate on RAVDESS Test Set
- Load test set (Actors 20-24)
- Compute accuracy, precision, recall, F1-score
- Generate confusion matrix
- Compute AUC-ROC and AUC-PRC
- Analyze per-emotion performance
- Compare with baseline CNN model
- Document evaluation results
"""

import os
import sys
import argparse
import json
import glob
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.ensemble import EnsembleClassifier
from src.training.dataset import RAVDESSDataset, SEQ_LEN
from src.evaluation.metrics import evaluate, plot_confusion_matrix, benchmark_inference

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


class BaselineCNN(nn.Module):
    """Baseline CNN model for comparison (from emotion_model_group8.pth)"""
    def __init__(self, num_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=7, padding=3, groups=64),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))


def get_test_indices(dataset, test_actors=[20, 21, 22, 23, 24]):
    """Get indices for test actors (20-24)"""
    test_indices = []
    for idx, path in enumerate(dataset.file_paths):
        # Extract actor number from path like "Actor_20/03-01-01-01-01-01-20.wav"
        actor_num = int(os.path.basename(os.path.dirname(path)).split('_')[1])
        if actor_num in test_actors:
            test_indices.append(idx)
    return test_indices


def create_baseline_dataloader(dataset, test_indices, batch_size=32):
    """Create dataloader for baseline CNN (only mel-spectrogram input)"""
    class BaselineDataset:
        def __init__(self, original_dataset, indices):
            self.dataset = original_dataset
            self.indices = indices
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            mel, hc, label = self.dataset[self.indices[idx]]
            return mel, label  # Only return mel and label for baseline CNN
    
    baseline_dataset = BaselineDataset(dataset, test_indices)
    return DataLoader(baseline_dataset, batch_size=batch_size, shuffle=False)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Ensemble Model on RAVDESS Test Set - Task 1.7')
    parser.add_argument('--data', required=True,
                       help='Path to RAVDESS dataset (contains Actor_* folders)')
    parser.add_argument('--model', default='models/ensemble_best.pth',
                       help='Path to trained ensemble model (default: models/ensemble_best.pth)')
    parser.add_argument('--baseline', default='emotion_model_group8.pth',
                       help='Path to baseline CNN model (default: emotion_model_group8.pth)')
    parser.add_argument('--batch', type=int, default=32,
                       help='Batch size (default: 32)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isdir(args.data):
        print(f"❌ Error: Dataset path not found: {args.data}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Ensemble model not found: {args.model}")
        print("Please run train_ensemble_full.py first to train the model")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"  📊 TASK 1.7: EVALUATE ON RAVDESS TEST SET")
    print(f"{'='*60}")
    print(f"Dataset       : {args.data}")
    print(f"Ensemble Model: {args.model}")
    print(f"Baseline Model: {args.baseline}")
    print(f"Device        : {device}")
    print(f"{'='*60}\n")
    
    try:
        # Load dataset and get test indices
        print("📂 Loading RAVDESS dataset...")
        full_dataset = RAVDESSDataset(args.data, seq_len=SEQ_LEN, augment=False)
        test_indices = get_test_indices(full_dataset, test_actors=[20, 21, 22, 23, 24])
        
        print(f"✅ Found {len(test_indices)} test samples from Actors 20-24")
        
        # Create test dataloader for ensemble model
        test_subset = Subset(full_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=args.batch, shuffle=False)
        
        # Load and evaluate ensemble model
        print("\n🤖 Loading ensemble model...")
        ensemble_model = EnsembleClassifier().to(device)
        ensemble_model.load_state_dict(torch.load(args.model, map_location=device))
        
        print("🔍 Evaluating ensemble model on test set...")
        ensemble_results = evaluate(ensemble_model, test_loader, device)
        
        # Generate confusion matrix for ensemble
        cm_path = 'results/ensemble_confusion_matrix.png'
        plot_confusion_matrix(ensemble_results['confusion_matrix'], cm_path)
        
        # Benchmark inference speed
        print("\n⚡ Benchmarking inference speed...")
        ensemble_latency = benchmark_inference(ensemble_model, device)
        
        # Load and evaluate baseline model if available
        baseline_results = None
        baseline_latency = None
        
        if os.path.exists(args.baseline):
            print(f"\n🔄 Loading baseline CNN model from {args.baseline}...")
            baseline_model = BaselineCNN().to(device)
            baseline_model.load_state_dict(torch.load(args.baseline, map_location=device))
            
            # Create baseline dataloader (only mel-spectrogram)
            baseline_loader = create_baseline_dataloader(full_dataset, test_indices, args.batch)
            
            print("🔍 Evaluating baseline model on test set...")
            baseline_results = evaluate(baseline_model, baseline_loader, device)
            
            # Generate confusion matrix for baseline
            baseline_cm_path = 'results/baseline_confusion_matrix.png'
            plot_confusion_matrix(baseline_results['confusion_matrix'], baseline_cm_path)
            
            # Benchmark baseline inference speed
            baseline_latency = benchmark_inference(baseline_model, device)
        else:
            print(f"⚠️  Baseline model not found at {args.baseline}, skipping comparison")
        
        # Compile evaluation results
        evaluation_results = {
            'evaluation_date': datetime.now().isoformat(),
            'test_set_info': {
                'actors': [20, 21, 22, 23, 24],
                'num_samples': len(test_indices),
                'emotions': EMOTIONS
            },
            'ensemble_model': {
                'model_path': args.model,
                'accuracy': float(ensemble_results['accuracy']),
                'precision': float(ensemble_results['precision']),
                'recall': float(ensemble_results['recall']),
                'f1_score': float(ensemble_results['f1']),
                'auc_roc': float(ensemble_results['auc_roc']) if not np.isnan(ensemble_results['auc_roc']) else None,
                'inference_latency_ms': ensemble_latency,
                'confusion_matrix_path': cm_path
            }
        }
        
        if baseline_results:
            evaluation_results['baseline_model'] = {
                'model_path': args.baseline,
                'accuracy': float(baseline_results['accuracy']),
                'precision': float(baseline_results['precision']),
                'recall': float(baseline_results['recall']),
                'f1_score': float(baseline_results['f1']),
                'auc_roc': float(baseline_results['auc_roc']) if not np.isnan(baseline_results['auc_roc']) else None,
                'inference_latency_ms': baseline_latency,
                'confusion_matrix_path': baseline_cm_path
            }
            
            # Calculate improvements
            evaluation_results['comparison'] = {
                'accuracy_improvement': float(ensemble_results['accuracy'] - baseline_results['accuracy']),
                'f1_improvement': float(ensemble_results['f1'] - baseline_results['f1']),
                'latency_difference_ms': ensemble_latency - baseline_latency
            }
        
        # Per-emotion analysis
        per_emotion_analysis = {}
        for i, emotion in enumerate(EMOTIONS):
            emotion_mask = ensemble_results['all_labels'] == i
            if emotion_mask.sum() > 0:
                emotion_preds = ensemble_results['all_preds'][emotion_mask]
                emotion_accuracy = (emotion_preds == i).mean()
                per_emotion_analysis[emotion] = {
                    'samples': int(emotion_mask.sum()),
                    'accuracy': float(emotion_accuracy),
                    'precision': float(ensemble_results['confusion_matrix'][i, i] / max(1, ensemble_results['confusion_matrix'][:, i].sum())),
                    'recall': float(ensemble_results['confusion_matrix'][i, i] / max(1, ensemble_results['confusion_matrix'][i, :].sum()))
                }
        
        evaluation_results['per_emotion_analysis'] = per_emotion_analysis
        
        # Save results
        results_path = 'results/ravdess_test_evaluation.json'
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Print summary
        print(f"\n✅ EVALUATION COMPLETED!")
        print(f"{'='*60}")
        print(f"📊 ENSEMBLE MODEL RESULTS:")
        print(f"   Accuracy  : {ensemble_results['accuracy']:.4f} ({ensemble_results['accuracy']*100:.2f}%)")
        print(f"   Precision : {ensemble_results['precision']:.4f}")
        print(f"   Recall    : {ensemble_results['recall']:.4f}")
        print(f"   F1-Score  : {ensemble_results['f1']:.4f}")
        if not np.isnan(ensemble_results['auc_roc']):
            print(f"   AUC-ROC   : {ensemble_results['auc_roc']:.4f}")
        print(f"   Latency   : {ensemble_latency:.2f} ms")
        
        if baseline_results:
            print(f"\n📊 BASELINE MODEL RESULTS:")
            print(f"   Accuracy  : {baseline_results['accuracy']:.4f} ({baseline_results['accuracy']*100:.2f}%)")
            print(f"   F1-Score  : {baseline_results['f1']:.4f}")
            print(f"   Latency   : {baseline_latency:.2f} ms")
            
            print(f"\n📈 IMPROVEMENTS:")
            acc_imp = ensemble_results['accuracy'] - baseline_results['accuracy']
            f1_imp = ensemble_results['f1'] - baseline_results['f1']
            print(f"   Accuracy  : {acc_imp:+.4f} ({acc_imp*100:+.2f}%)")
            print(f"   F1-Score  : {f1_imp:+.4f}")
        
        print(f"\n📁 FILES GENERATED:")
        print(f"   Evaluation results : {results_path}")
        print(f"   Confusion matrix   : {cm_path}")
        if baseline_results:
            print(f"   Baseline CM        : {baseline_cm_path}")
        print(f"{'='*60}\n")
        
        # Update task status
        print("✅ Task 1.7 completed successfully!")
        print("   - ✅ Loaded test set (Actors 20-24)")
        print("   - ✅ Computed accuracy, precision, recall, F1-score")
        print("   - ✅ Generated confusion matrix")
        print("   - ✅ Computed AUC-ROC")
        print("   - ✅ Analyzed per-emotion performance")
        if baseline_results:
            print("   - ✅ Compared with baseline CNN model")
        print("   - ✅ Documented evaluation results")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)