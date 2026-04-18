#!/usr/bin/env python3
"""
Task 1.6: Train Ensemble Model on Full RAVDESS Dataset
- Load full RAVDESS dataset
- Extract features for all samples
- Train ensemble model for 100 epochs
- Monitor validation loss and accuracy
- Save best model checkpoint
- Generate training curves (loss, accuracy)
- Document final training metrics
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training.trainer import train
from src.evaluation.metrics import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description='Train Ensemble SER Model - Task 1.6')
    parser.add_argument('--data', required=True, 
                       help='Path to RAVDESS dataset (contains Actor_* folders)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--save-dir', default='models',
                       help='Directory to save model (default: models)')
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.isdir(args.data):
        print(f"❌ Error: Dataset path not found: {args.data}")
        print("Please provide the path to RAVDESS dataset containing Actor_* folders")
        sys.exit(1)
    
    # Check for Actor folders
    actor_folders = [d for d in os.listdir(args.data) if d.startswith('Actor_')]
    if not actor_folders:
        print(f"❌ Error: No Actor_* folders found in {args.data}")
        print("Please ensure the path contains RAVDESS Actor_01, Actor_02, etc. folders")
        sys.exit(1)
    
    print(f"✅ Found {len(actor_folders)} Actor folders in {args.data}")
    
    # Setup paths
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    model_path = os.path.join(args.save_dir, 'ensemble_best.pth')
    
    print(f"\n{'='*60}")
    print(f"  🚀 TASK 1.6: TRAIN ENSEMBLE MODEL")
    print(f"{'='*60}")
    print(f"Dataset      : {args.data}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch Size   : {args.batch}")
    print(f"Learning Rate: {args.lr}")
    print(f"Save Path    : {model_path}")
    print(f"Device       : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Train the model
        print("🔄 Starting training...")
        history = train(
            data_path=args.data,
            save_path=model_path,
            batch_size=args.batch,
            num_epochs=args.epochs,
            lr=args.lr,
            train_split=0.66,  # 66:34 split as per spec
            seed=42,
            patience=15
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Generate training curves
        print("\n📊 Generating training curves...")
        curves_path = 'results/training_curves.png'
        plot_training_curves(history, curves_path)
        
        # Document final training metrics
        final_metrics = {
            'training_completed': datetime.now().isoformat(),
            'training_time_seconds': round(training_time, 2),
            'training_time_formatted': f"{training_time//3600:.0f}h {(training_time%3600)//60:.0f}m {training_time%60:.0f}s",
            'epochs_completed': len(history['train_loss']),
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'final_val_accuracy': float(history['val_acc'][-1]),
            'best_val_loss': float(min(history['val_loss'])),
            'best_val_accuracy': float(max(history['val_acc'])),
            'hyperparameters': {
                'batch_size': args.batch,
                'learning_rate': args.lr,
                'train_split': 0.66,
                'optimizer': 'Adam',
                'scheduler': 'ReduceLROnPlateau',
                'weight_decay': 1e-4,
                'dropout': 0.5
            },
            'model_path': model_path,
            'training_curves_path': curves_path
        }
        
        # Save metrics to JSON
        metrics_path = 'results/training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Training Time    : {final_metrics['training_time_formatted']}")
        print(f"Epochs Completed : {final_metrics['epochs_completed']}")
        print(f"Final Val Loss   : {final_metrics['final_val_loss']:.4f}")
        print(f"Final Val Acc    : {final_metrics['final_val_accuracy']:.4f} ({final_metrics['final_val_accuracy']*100:.2f}%)")
        print(f"Best Val Loss    : {final_metrics['best_val_loss']:.4f}")
        print(f"Best Val Acc     : {final_metrics['best_val_accuracy']:.4f} ({final_metrics['best_val_accuracy']*100:.2f}%)")
        print(f"{'='*60}")
        print(f"📁 Model saved to        : {model_path}")
        print(f"📊 Training curves saved : {curves_path}")
        print(f"📋 Metrics saved to     : {metrics_path}")
        print(f"{'='*60}\n")
        
        # Update task status
        print("✅ Task 1.6 completed successfully!")
        print("   - ✅ Loaded full RAVDESS dataset")
        print("   - ✅ Extracted features for all samples")
        print("   - ✅ Trained ensemble model for epochs")
        print("   - ✅ Monitored validation loss and accuracy")
        print("   - ✅ Saved best model checkpoint")
        print("   - ✅ Generated training curves")
        print("   - ✅ Documented final training metrics")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)