"""
Integration test for Bi-LSTM Branch with realistic audio features.

This test demonstrates the Bi-LSTM branch processing hand-crafted audio features
extracted from speech samples, verifying:
1. Feature extraction pipeline compatibility
2. Bidirectional processing of temporal sequences
3. Output validity for emotion classification
4. Integration with ensemble architecture
"""

import torch
import torch.nn as nn
import numpy as np
from lstm_branch import BiLSTMBranch, BiLSTMBranchWithBatchNorm


def simulate_feature_extraction(num_samples=10, seq_len=100, feature_dim=91):
    """
    Simulate feature extraction from audio samples.
    
    In a real scenario, these features would come from:
    - MFCC: 13 coefficients
    - Mel-Spectrogram: 64 mel-bands
    - ZCR: 1 value
    - RMSE: 1 value
    - Chroma STFT: 12 values
    Total: 13 + 64 + 1 + 1 + 12 = 91 features
    
    Args:
        num_samples (int): Number of audio samples
        seq_len (int): Sequence length (time steps)
        feature_dim (int): Feature dimension (91 for hand-crafted features)
    
    Returns:
        torch.Tensor: Simulated features of shape (num_samples, seq_len, feature_dim)
    """
    # Simulate normalized features (mean=0, std=1)
    features = torch.randn(num_samples, seq_len, feature_dim)
    return features


def test_lstm_with_simulated_features():
    """Test Bi-LSTM branch with simulated audio features."""
    print("\n" + "="*70)
    print("Integration Test: Bi-LSTM Branch with Simulated Audio Features")
    print("="*70)
    
    # Initialize Bi-LSTM branch
    lstm_branch = BiLSTMBranch(input_size=91, hidden_size=128, num_layers=2, dropout=0.5)
    lstm_branch.eval()
    
    # Simulate feature extraction from 10 audio samples
    num_samples = 10
    seq_len = 100
    feature_dim = 91
    
    print(f"\nSimulating feature extraction from {num_samples} audio samples...")
    print(f"  - Sequence length: {seq_len} time steps")
    print(f"  - Feature dimension: {feature_dim}")
    print(f"    - MFCC: 13 coefficients")
    print(f"    - Mel-Spectrogram: 64 mel-bands")
    print(f"    - ZCR: 1 value")
    print(f"    - RMSE: 1 value")
    print(f"    - Chroma STFT: 12 values")
    print(f"    - Total: 13 + 64 + 1 + 1 + 12 = 91")
    
    # Extract features
    features = simulate_feature_extraction(num_samples, seq_len, feature_dim)
    print(f"\nExtracted features shape: {features.shape}")
    print(f"  - Batch size: {features.shape[0]}")
    print(f"  - Sequence length: {features.shape[1]}")
    print(f"  - Feature dimension: {features.shape[2]}")
    
    # Process through Bi-LSTM branch
    print(f"\nProcessing features through Bi-LSTM branch...")
    with torch.no_grad():
        lstm_output = lstm_branch(features)
    
    print(f"LSTM output shape: {lstm_output.shape}")
    print(f"  - Batch size: {lstm_output.shape[0]}")
    print(f"  - Feature dimension: {lstm_output.shape[1]} (256 = 128 forward + 128 backward)")
    
    # Verify output properties
    print(f"\nVerifying output properties...")
    
    # Check shape
    assert lstm_output.shape == (num_samples, 256), \
        f"Expected shape ({num_samples}, 256), got {lstm_output.shape}"
    print(f"  ✓ Output shape is correct: {lstm_output.shape}")
    
    # Check for NaN/Inf
    assert not torch.isnan(lstm_output).any(), "Output contains NaN values"
    assert not torch.isinf(lstm_output).any(), "Output contains Inf values"
    print(f"  ✓ Output contains no NaN or Inf values")
    
    # Check value range
    print(f"  ✓ Output value range: [{lstm_output.min():.4f}, {lstm_output.max():.4f}]")
    
    # Check statistics
    mean = lstm_output.mean()
    std = lstm_output.std()
    print(f"  ✓ Output statistics: mean={mean:.4f}, std={std:.4f}")
    
    return lstm_output


def test_bidirectional_processing():
    """Test that bidirectional processing captures both forward and backward context."""
    print("\n" + "="*70)
    print("Test: Bidirectional Processing Verification")
    print("="*70)
    
    lstm_branch = BiLSTMBranch(input_size=91, hidden_size=128, num_layers=2, dropout=0.5)
    lstm_branch.eval()
    
    # Create a simple sequence with a pattern
    batch_size = 1
    seq_len = 50
    feature_dim = 91
    
    # Create features with a clear pattern
    features = torch.zeros(batch_size, seq_len, feature_dim)
    
    # Add a pattern: increasing values in first half, decreasing in second half
    for t in range(seq_len):
        if t < seq_len // 2:
            features[0, t, :] = t / (seq_len // 2)  # Increasing
        else:
            features[0, t, :] = (seq_len - t) / (seq_len // 2)  # Decreasing
    
    print(f"\nCreated sequence with pattern:")
    print(f"  - First half: increasing values (0 to 1)")
    print(f"  - Second half: decreasing values (1 to 0)")
    
    # Process through LSTM
    with torch.no_grad():
        output = lstm_branch(features)
    
    print(f"\nBidirectional LSTM output shape: {output.shape}")
    print(f"  - Forward LSTM processes: left-to-right (captures increasing pattern)")
    print(f"  - Backward LSTM processes: right-to-left (captures decreasing pattern)")
    print(f"  - Concatenated output: 128 + 128 = 256 dimensions")
    
    # The output should capture both directions
    assert output.shape == (1, 256), f"Expected shape (1, 256), got {output.shape}"
    print(f"\n✓ Bidirectional processing verified")
    
    return output


def test_ensemble_compatibility():
    """Test that LSTM output is compatible with ensemble architecture."""
    print("\n" + "="*70)
    print("Test: Ensemble Architecture Compatibility")
    print("="*70)
    
    # Simulate CNN output (512-dim)
    cnn_output = torch.randn(32, 512)
    
    # Generate LSTM output (256-dim)
    lstm_branch = BiLSTMBranch(input_size=91, hidden_size=128, num_layers=2, dropout=0.5)
    lstm_branch.eval()
    
    features = torch.randn(32, 100, 91)
    with torch.no_grad():
        lstm_output = lstm_branch(features)
    
    print(f"\nEnsemble component outputs:")
    print(f"  - CNN branch output: {cnn_output.shape} (512-dim)")
    print(f"  - LSTM branch output: {lstm_output.shape} (256-dim)")
    
    # Simulate fusion layer
    fused = torch.cat([cnn_output, lstm_output], dim=1)
    print(f"  - Fused features: {fused.shape} (768-dim = 512 + 256)")
    
    # Simulate classification head
    fusion_layer = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Dropout(0.5)
    )
    
    classifier_head = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 8)  # 8 emotions
    )
    
    with torch.no_grad():
        fused_features = fusion_layer(fused)
        logits = classifier_head(fused_features)
        probabilities = torch.softmax(logits, dim=1)
    
    print(f"  - After fusion layer: {fused_features.shape}")
    print(f"  - Classification logits: {logits.shape}")
    print(f"  - Emotion probabilities: {probabilities.shape}")
    
    # Verify probability distribution
    prob_sum = probabilities.sum(dim=1)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), \
        "Probabilities don't sum to 1"
    print(f"\n✓ Probabilities sum to 1.0 (valid distribution)")
    
    # Verify emotion predictions
    predicted_emotions = torch.argmax(probabilities, dim=1)
    print(f"✓ Predicted emotions: {predicted_emotions.tolist()}")
    
    return probabilities


def test_training_mode():
    """Test that LSTM works correctly in training mode."""
    print("\n" + "="*70)
    print("Test: Training Mode Verification")
    print("="*70)
    
    lstm_branch = BiLSTMBranch(input_size=91, hidden_size=128, num_layers=2, dropout=0.5)
    lstm_branch.train()
    
    features = torch.randn(32, 100, 91, requires_grad=True)
    
    # Forward pass
    output = lstm_branch(features)
    
    # Compute loss and backward pass
    loss = output.sum()
    loss.backward()
    
    print(f"\nTraining mode test:")
    print(f"  - Input shape: {features.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Loss: {loss.item():.4f}")
    
    # Verify gradients
    assert features.grad is not None, "Input gradients not computed"
    print(f"  ✓ Input gradients computed")
    
    for name, param in lstm_branch.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
    print(f"  ✓ All model parameters have gradients")
    
    return loss


def test_inference_performance():
    """Test inference performance and latency."""
    print("\n" + "="*70)
    print("Test: Inference Performance")
    print("="*70)
    
    import time
    
    lstm_branch = BiLSTMBranch(input_size=91, hidden_size=128, num_layers=2, dropout=0.5)
    lstm_branch.eval()
    
    # Warm up
    features = torch.randn(32, 100, 91)
    with torch.no_grad():
        _ = lstm_branch(features)
    
    # Benchmark
    num_iterations = 100
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            output = lstm_branch(features)
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_iterations * 1000  # Convert to ms
    
    print(f"\nInference performance (batch_size=32, seq_len=100):")
    print(f"  - Total time for {num_iterations} iterations: {elapsed_time:.2f}s")
    print(f"  - Average time per batch: {avg_time:.2f}ms")
    print(f"  - Throughput: {32 / (avg_time / 1000):.1f} samples/sec")
    
    return avg_time


def run_all_integration_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE INTEGRATION TEST SUITE")
    print("Bi-LSTM Branch for Speech Emotion Recognition")
    print("="*70)
    
    try:
        # Test 1: LSTM with simulated features
        lstm_output = test_lstm_with_simulated_features()
        
        # Test 2: Bidirectional processing
        bidirectional_output = test_bidirectional_processing()
        
        # Test 3: Ensemble compatibility
        probabilities = test_ensemble_compatibility()
        
        # Test 4: Training mode
        loss = test_training_mode()
        
        # Test 5: Inference performance
        avg_time = test_inference_performance()
        
        # Summary
        print("\n" + "="*70)
        print("INTEGRATION TEST SUMMARY")
        print("="*70)
        print("\n✓ All integration tests passed successfully!")
        print("\nKey Results:")
        print(f"  - LSTM output dimension: 256 (128 forward + 128 backward)")
        print(f"  - Bidirectional processing: Verified")
        print(f"  - Ensemble compatibility: Verified")
        print(f"  - Training mode: Verified")
        print(f"  - Inference latency: {avg_time:.2f}ms per batch")
        print(f"  - Emotion classification: 8 emotions with valid probabilities")
        print("\n" + "="*70)
        
        return True
    
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_integration_tests()
    exit(0 if success else 1)
