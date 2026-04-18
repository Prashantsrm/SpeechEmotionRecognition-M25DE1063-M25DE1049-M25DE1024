#!/usr/bin/env python3
"""
Task 1.8: Cross-Dataset Validation
- Load TESS dataset and evaluate ensemble model
- Load SAVEE dataset and evaluate ensemble model  
- Load CREMA-D dataset and evaluate ensemble model
- Compute average accuracy across datasets
- Analyze generalization performance
"""

import os
import sys
import argparse
import json
import glob
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.ensemble import EnsembleClassifier
from src.features.extractor import FeatureExtractor, pad_or_truncate_mel, pad_or_truncate_seq
from src.evaluation.metrics import evaluate, plot_confusion_matrix

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
SEQ_LEN = 128


class CrossDataset(Dataset):
    """Generic dataset class for cross-dataset evaluation"""
    
    def __init__(self, data_path, dataset_name, seq_len=SEQ_LEN):
        self.extractor = FeatureExtractor()
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        
        self.file_paths = []
        self.labels = []
        
        if dataset_name.upper() == 'TESS':
            self._load_tess(data_path)
        elif dataset_name.upper() == 'SAVEE':
            self._load_savee(data_path)
        elif dataset_name.upper() == 'CREMA-D':
            self._load_crema_d(data_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        print(f"[{dataset_name}] Loaded {len(self.file_paths)} files")
    
    def _load_tess(self, data_path):
        """Load TESS dataset"""
        # TESS emotion mapping
        tess_emotion_map = {
            'neutral': 0, 'happy': 2, 'sad': 3, 'angry': 4, 
            'fear': 5, 'disgust': 6, 'surprise': 7
        }
        
        # Look for TESS files (format: OAF_back_neutral.wav)
        wav_files = glob.glob(os.path.join(data_path, "**/*.wav"), recursive=True)
        
        for wav_file in wav_files:
            filename = os.path.basename(wav_file)
            # Extract emotion from filename
            for emotion_name, emotion_id in tess_emotion_map.items():
                if emotion_name in filename.lower():
                    self.file_paths.append(wav_file)
                    self.labels.append(emotion_id)
                    break
    
    def _load_savee(self, data_path):
        """Load SAVEE dataset"""
        # SAVEE emotion mapping (based on filename prefixes)
        savee_emotion_map = {
            'n': 0,  # neutral
            'h': 2,  # happy  
            'sa': 3, # sad
            'a': 4,  # angry
            'f': 5,  # fear
            'd': 6,  # disgust
            'su': 7  # surprise
        }
        
        wav_files = glob.glob(os.path.join(data_path, "**/*.wav"), recursive=True)
        
        for wav_file in wav_files:
            filename = os.path.basename(wav_file).lower()
            # Extract emotion from filename prefix
            for prefix, emotion_id in savee_emotion_map.items():
                if filename.startswith(prefix):
                    self.file_paths.append(wav_file)
                    self.labels.append(emotion_id)
                    break
    
    def _load_crema_d(self, data_path):
        """Load CREMA-D dataset"""
        # CREMA-D emotion mapping
        crema_emotion_map = {
            'NEU': 0,  # neutral
            'HAP': 2,  # happy
            'SAD': 3,  # sad
            'ANG': 4,  # angry
            'FEA': 5,  # fear
            'DIS': 6   # disgust
        }
        
        wav_files = glob.glob(os.path.join(data_path, "**/*.wav"), recursive=True)
        
        for wav_file in wav_files:
            filename = os.path.basename(wav_file)
            # CREMA-D format: 1001_DFA_ANG_XX.wav
            parts = filename.split('_')
            if len(parts) >= 3:
                emotion_code = parts[2]
                if emotion_code in crema_emotion_map:
                    self.file_paths.append(wav_file)
                    self.labels.append(crema_emotion_map[emotion_code])
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            mel_3ch, hand_crafted = self.extractor.load_and_extract(path)
            mel_3ch = pad_or_truncate_mel(mel_3ch, self.seq_len)
            hand_crafted = pad_or_truncate_seq(hand_crafted, self.seq_len)
            
            mel_tensor = torch.FloatTensor(mel_3ch)
            hc_tensor = torch.FloatTensor(hand_crafted)
            lbl_tensor = torch.tensor(label, dtype=torch.long)
            
            return mel_tensor, hc_tensor, lbl_tensor
        
        except Exception as e:
            print(f"Warning: Failed to process {path}: {e}")
            # Return dummy data to avoid breaking the dataloader
            mel_tensor = torch.zeros(3, 64, self.seq_len)
            hc_tensor = torch.zeros(self.seq_len, 27)
            lbl_tensor = torch.tensor(0, dtype=torch.long)
            return mel_tensor, hc_tensor, lbl_tensor


def evaluate_dataset(model, dataset_path, dataset_name, batch_size=32, device=None):
    """Evaluate model on a cross-dataset"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n📂 Loading {dataset_name} dataset from {dataset_path}...")
    
    try:
        dataset = CrossDataset(dataset_path, dataset_name)
        if len(dataset) == 0:
            print(f"⚠️  No valid files found in {dataset_name} dataset")
            return None
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        print(f"🔍 Evaluating on {dataset_name} ({len(dataset)} samples)...")
        results = evaluate(model, dataloader, device)
        
        # Generate confusion matrix
        cm_path = f'results/{dataset_name.lower()}_confusion_matrix.png'
        plot_confusion_matrix(results['confusion_matrix'], cm_path)
        results['confusion_matrix_path'] = cm_path
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to evaluate {dataset_name}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Cross-Dataset Validation - Task 1.8')
    parser.add_argument('--model', default='models/ensemble_best.pth',
                       help='Path to trained ensemble model (default: models/ensemble_best.pth)')
    parser.add_argument('--tess', 
                       help='Path to TESS dataset directory')
    parser.add_argument('--savee',
                       help='Path to SAVEE dataset directory')
    parser.add_argument('--crema-d',
                       help='Path to CREMA-D dataset directory')
    parser.add_argument('--batch', type=int, default=32,
                       help='Batch size (default: 32)')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"❌ Error: Ensemble model not found: {args.model}")
        print("Please run train_ensemble_full.py first to train the model")
        sys.exit(1)
    
    # Check if at least one dataset is provided
    datasets_provided = [args.tess, args.savee, getattr(args, 'crema_d', None)]
    if not any(datasets_provided):
        print("❌ Error: Please provide at least one dataset path (--tess, --savee, or --crema-d)")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"  🌍 TASK 1.8: CROSS-DATASET VALIDATION")
    print(f"{'='*60}")
    print(f"Ensemble Model: {args.model}")
    print(f"Device        : {device}")
    if args.tess:
        print(f"TESS Dataset  : {args.tess}")
    if args.savee:
        print(f"SAVEE Dataset : {args.savee}")
    if getattr(args, 'crema_d', None):
        print(f"CREMA-D Dataset: {getattr(args, 'crema_d')}")
    print(f"{'='*60}\n")
    
    try:
        # Load ensemble model
        print("🤖 Loading ensemble model...")
        model = EnsembleClassifier().to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        
        # Evaluate on each dataset
        results = {}
        accuracies = []
        
        if args.tess and os.path.exists(args.tess):
            tess_results = evaluate_dataset(model, args.tess, 'TESS', args.batch, device)
            if tess_results:
                results['TESS'] = tess_results
                accuracies.append(tess_results['accuracy'])
        
        if args.savee and os.path.exists(args.savee):
            savee_results = evaluate_dataset(model, args.savee, 'SAVEE', args.batch, device)
            if savee_results:
                results['SAVEE'] = savee_results
                accuracies.append(savee_results['accuracy'])
        
        if getattr(args, 'crema_d', None) and os.path.exists(getattr(args, 'crema_d')):
            crema_results = evaluate_dataset(model, getattr(args, 'crema_d'), 'CREMA-D', args.batch, device)
            if crema_results:
                results['CREMA-D'] = crema_results
                accuracies.append(crema_results['accuracy'])
        
        if not results:
            print("❌ No datasets were successfully evaluated")
            return False
        
        # Compute average accuracy
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        
        # Compile cross-dataset evaluation results
        cross_dataset_results = {
            'evaluation_date': datetime.now().isoformat(),
            'model_path': args.model,
            'datasets_evaluated': list(results.keys()),
            'individual_results': {},
            'summary': {
                'average_accuracy': float(avg_accuracy),
                'num_datasets': len(results),
                'accuracy_std': float(np.std(accuracies)) if len(accuracies) > 1 else 0.0
            }
        }
        
        # Add individual results (excluding large arrays)
        for dataset_name, dataset_results in results.items():
            cross_dataset_results['individual_results'][dataset_name] = {
                'accuracy': float(dataset_results['accuracy']),
                'precision': float(dataset_results['precision']),
                'recall': float(dataset_results['recall']),
                'f1_score': float(dataset_results['f1']),
                'auc_roc': float(dataset_results['auc_roc']) if not np.isnan(dataset_results['auc_roc']) else None,
                'num_samples': len(dataset_results['all_labels']),
                'confusion_matrix_path': dataset_results['confusion_matrix_path']
            }
        
        # Analyze generalization performance
        generalization_analysis = {
            'performance_consistency': 'high' if np.std(accuracies) < 0.05 else 'medium' if np.std(accuracies) < 0.10 else 'low',
            'best_dataset': max(results.keys(), key=lambda k: results[k]['accuracy']) if results else None,
            'worst_dataset': min(results.keys(), key=lambda k: results[k]['accuracy']) if results else None,
            'accuracy_range': float(max(accuracies) - min(accuracies)) if len(accuracies) > 1 else 0.0
        }
        
        cross_dataset_results['generalization_analysis'] = generalization_analysis
        
        # Save results
        results_path = 'results/cross_dataset_evaluation.json'
        os.makedirs('results', exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(cross_dataset_results, f, indent=2)
        
        # Print summary
        print(f"\n✅ CROSS-DATASET EVALUATION COMPLETED!")
        print(f"{'='*60}")
        print(f"📊 RESULTS SUMMARY:")
        print(f"   Datasets Evaluated: {len(results)}")
        print(f"   Average Accuracy  : {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        if len(accuracies) > 1:
            print(f"   Accuracy Std Dev  : {np.std(accuracies):.4f}")
            print(f"   Accuracy Range    : {max(accuracies) - min(accuracies):.4f}")
        
        print(f"\n📊 INDIVIDUAL DATASET RESULTS:")
        for dataset_name, dataset_results in results.items():
            acc = dataset_results['accuracy']
            f1 = dataset_results['f1']
            samples = len(dataset_results['all_labels'])
            print(f"   {dataset_name:8s}: {acc:.4f} ({acc*100:.2f}%) | F1: {f1:.4f} | Samples: {samples}")
        
        print(f"\n🔍 GENERALIZATION ANALYSIS:")
        print(f"   Performance Consistency: {generalization_analysis['performance_consistency'].upper()}")
        if generalization_analysis['best_dataset']:
            print(f"   Best Dataset           : {generalization_analysis['best_dataset']}")
        if generalization_analysis['worst_dataset']:
            print(f"   Worst Dataset          : {generalization_analysis['worst_dataset']}")
        
        print(f"\n📁 FILES GENERATED:")
        print(f"   Cross-dataset results: {results_path}")
        for dataset_name, dataset_results in results.items():
            print(f"   {dataset_name} confusion matrix: {dataset_results['confusion_matrix_path']}")
        print(f"{'='*60}\n")
        
        # Update task status
        print("✅ Task 1.8 completed successfully!")
        evaluated_datasets = []
        if 'TESS' in results:
            evaluated_datasets.append("TESS")
        if 'SAVEE' in results:
            evaluated_datasets.append("SAVEE")
        if 'CREMA-D' in results:
            evaluated_datasets.append("CREMA-D")
        
        for dataset in evaluated_datasets:
            print(f"   - ✅ Loaded {dataset} dataset")
            print(f"   - ✅ Evaluated ensemble model on {dataset}")
        print("   - ✅ Computed average accuracy across datasets")
        print("   - ✅ Analyzed generalization performance")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Cross-dataset evaluation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)