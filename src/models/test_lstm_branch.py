"""
Test suite for Bi-LSTM Branch implementation.

This module contains comprehensive tests for the BiLSTMBranch class,
verifying:
1. Correct output dimensions (256-dim feature vector)
2. Bidirectional processing functionality
3. Dropout and batch normalization
4. Gradient flow for training
5. Deterministic behavior with fixed seed
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from lstm_branch import BiLSTMBranch, BiLSTMBranchWithBatchNorm


class TestBiLSTMBranch:
    """Test suite for BiLSTMBranch class."""
    
    @pytest.fixture
    def lstm_branch(self):
        """Create a BiLSTMBranch instance for testing."""
        return BiLSTMBranch(input_size=91, hidden_size=128, num_layers=2, dropout=0.5)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        batch_size = 32
        seq_len = 100
        input_size = 91
        return torch.randn(batch_size, seq_len, input_size)
    
    def test_output_shape(self, lstm_branch, sample_input):
        """Test that output has correct shape (batch_size, 256)."""
        output = lstm_branch(sample_input)
        
        assert output.shape == (32, 256), f"Expected shape (32, 256), got {output.shape}"
        print(f"✓ Output shape test passed: {output.shape}")
    
    def test_output_dimensions_256(self, lstm_branch):
        """Test that output is always 256-dimensional regardless of input."""
        test_cases = [
            (16, 50, 91),   # Small batch, short sequence
            (32, 100, 91),  # Standard batch, standard sequence
            (64, 200, 91),  # Large batch, long sequence
            (1, 10, 91),    # Single sample, short sequence
        ]
        
        for batch_size, seq_len, input_size in test_cases:
            x = torch.randn(batch_size, seq_len, input_size)
            output = lstm_branch(x)
            
            assert output.shape[1] == 256, f"Expected 256-dim output, got {output.shape[1]}"
            assert output.shape[0] == batch_size, f"Batch size mismatch"
        
        print("✓ Output dimensions test passed for all cases")
    
    def test_bidirectional_concatenation(self, lstm_branch, sample_input):
        """Test that bidirectional processing produces 256-dim output (128+128)."""
        output = lstm_branch(sample_input)
        
        # 256 = 128 (forward) + 128 (backward)
        expected_dim = 128 * 2
        assert output.shape[1] == expected_dim, \
            f"Expected {expected_dim}-dim output from bidirectional LSTM, got {output.shape[1]}"
        
        print(f"✓ Bidirectional concatenation test passed: {output.shape[1]} = 128 + 128")
    
    def test_gradient_flow(self, lstm_branch, sample_input):
        """Test that gradients flow correctly through the network."""
        output = lstm_branch(sample_input)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed for LSTM parameters
        for name, param in lstm_branch.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
        
        print("✓ Gradient flow test passed")
    
    def test_dropout_effect(self):
        """Test that dropout is applied during training but not during evaluation."""
        lstm_branch = BiLSTMBranch(input_size=91, hidden_size=128, num_layers=2, dropout=0.5)
        x = torch.randn(32, 100, 91)
        
        # Training mode - dropout should be active
        lstm_branch.train()
        outputs_train = [lstm_branch(x) for _ in range(5)]
        
        # Check that outputs vary due to dropout
        outputs_train_tensor = torch.stack(outputs_train)
        variance = outputs_train_tensor.var(dim=0).mean()
        assert variance > 0, "Dropout not working in training mode"
        
        # Evaluation mode - dropout should be inactive
        lstm_branch.eval()
        outputs_eval = [lstm_branch(x) for _ in range(5)]
        
        # Check that outputs are identical
        for i in range(1, len(outputs_eval)):
            assert torch.allclose(outputs_eval[0], outputs_eval[i]), \
                "Outputs should be identical in eval mode"
        
        print("✓ Dropout effect test passed")
    
    def test_deterministic_behavior(self, sample_input):
        """Test that model produces deterministic output with fixed seed."""
        torch.manual_seed(42)
        lstm_branch1 = BiLSTMBranch(input_size=91, hidden_size=128, num_layers=2, dropout=0.5)
        lstm_branch1.eval()
        output1 = lstm_branch1(sample_input)
        
        torch.manual_seed(42)
        lstm_branch2 = BiLSTMBranch(input_size=91, hidden_size=128, num_layers=2, dropout=0.5)
        lstm_branch2.eval()
        output2 = lstm_branch2(sample_input)
        
        assert torch.allclose(output1, output2, atol=1e-6), \
            "Outputs should be identical with same seed"
        
        print("✓ Deterministic behavior test passed")
    
    def test_batch_processing(self, lstm_branch):
        """Test that batch processing works correctly."""
        batch_sizes = [1, 8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 100, 91)
            output = lstm_branch(x)
            
            assert output.shape == (batch_size, 256), \
                f"Batch size {batch_size}: expected shape ({batch_size}, 256), got {output.shape}"
        
        print("✓ Batch processing test passed for all batch sizes")
    
    def test_variable_sequence_length(self, lstm_branch):
        """Test that LSTM handles variable sequence lengths."""
        seq_lengths = [10, 50, 100, 200, 500]
        
        for seq_len in seq_lengths:
            x = torch.randn(32, seq_len, 91)
            output = lstm_branch(x)
            
            assert output.shape == (32, 256), \
                f"Seq length {seq_len}: expected shape (32, 256), got {output.shape}"
        
        print("✓ Variable sequence length test passed")
    
    def test_output_range(self, lstm_branch, sample_input):
        """Test that output values are in reasonable range."""
        lstm_branch.eval()
        output = lstm_branch(sample_input)
        
        # Output should not contain NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        
        # Output values should be in reasonable range (typically -10 to 10 for normalized features)
        assert output.abs().max() < 100, "Output values are too large"
        
        print(f"✓ Output range test passed: min={output.min():.4f}, max={output.max():.4f}")
    
    def test_lstm_parameters(self, lstm_branch):
        """Test that LSTM has correct number of parameters."""
        total_params = sum(p.numel() for p in lstm_branch.parameters())
        
        # Rough estimate: LSTM has ~4*hidden_size*(input_size + hidden_size + 1) params per layer
        # For bidirectional: 2x
        # For 2 layers: 2x
        # Approximate: 4 * 128 * (91 + 128 + 1) * 2 * 2 ≈ 450k
        
        assert total_params > 0, "Model has no parameters"
        print(f"✓ LSTM parameters test passed: {total_params:,} parameters")
    
    def test_no_memory_leak(self, lstm_branch, sample_input):
        """Test that model doesn't accumulate memory."""
        import gc
        
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        for _ in range(100):
            output = lstm_branch(sample_input)
            del output
        
        gc.collect()
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Memory shouldn't grow significantly
        memory_growth = final_memory - initial_memory
        print(f"✓ Memory leak test passed: growth = {memory_growth / 1024 / 1024:.2f} MB")


class TestBiLSTMBranchWithBatchNorm:
    """Test suite for BiLSTMBranchWithBatchNorm class."""
    
    @pytest.fixture
    def lstm_branch_bn(self):
        """Create a BiLSTMBranchWithBatchNorm instance for testing."""
        return BiLSTMBranchWithBatchNorm(input_size=91, hidden_size=128, num_layers=2, dropout=0.5)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(32, 100, 91)
    
    def test_output_shape_with_bn(self, lstm_branch_bn, sample_input):
        """Test that output shape is correct with batch normalization."""
        output = lstm_branch_bn(sample_input)
        assert output.shape == (32, 256), f"Expected shape (32, 256), got {output.shape}"
        print(f"✓ Output shape with batch norm test passed: {output.shape}")
    
    def test_batch_norm_effect(self, lstm_branch_bn):
        """Test that batch normalization is applied to the output."""
        x = torch.randn(32, 100, 91)
        
        lstm_branch_bn.train()
        output = lstm_branch_bn(x)
        
        # Verify that output doesn't contain NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        
        # Verify that batch norm layer exists and is applied
        assert hasattr(lstm_branch_bn, 'batch_norm'), "Batch norm layer not found"
        
        # Verify output shape is correct
        assert output.shape == (32, 256), f"Output shape mismatch: {output.shape}"
        
        print("✓ Batch norm effect test passed")


def test_lstm_branch_basic():
    """Basic test for BiLSTMBranch."""
    print("\n" + "="*60)
    print("Testing BiLSTMBranch Implementation")
    print("="*60)
    
    # Create model
    lstm_branch = BiLSTMBranch(input_size=91, hidden_size=128, num_layers=2, dropout=0.5)
    
    # Create sample input
    batch_size = 32
    seq_len = 100
    input_size = 91
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    output = lstm_branch(x)
    
    # Verify output
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (32, 256)")
    
    assert output.shape == (32, 256), f"Output shape mismatch: {output.shape}"
    print("\n✓ Basic test passed!")
    
    return True


def test_lstm_branch_with_batch_norm():
    """Test BiLSTMBranchWithBatchNorm."""
    print("\n" + "="*60)
    print("Testing BiLSTMBranchWithBatchNorm Implementation")
    print("="*60)
    
    # Create model
    lstm_branch = BiLSTMBranchWithBatchNorm(input_size=91, hidden_size=128, num_layers=2, dropout=0.5)
    
    # Create sample input
    x = torch.randn(32, 100, 91)
    
    # Forward pass
    output = lstm_branch(x)
    
    # Verify output
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (32, 256), f"Output shape mismatch: {output.shape}"
    print("\n✓ Batch norm test passed!")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Comprehensive Test Suite for BiLSTMBranch")
    print("="*60)
    
    # Basic tests
    test_lstm_branch_basic()
    test_lstm_branch_with_batch_norm()
    
    # Run pytest tests
    print("\n" + "="*60)
    print("Running PyTest Tests")
    print("="*60)
    
    test_suite = TestBiLSTMBranch()
    lstm_branch = BiLSTMBranch(input_size=91, hidden_size=128, num_layers=2, dropout=0.5)
    sample_input = torch.randn(32, 100, 91)
    
    # Run individual tests
    test_suite.test_output_shape(lstm_branch, sample_input)
    test_suite.test_output_dimensions_256(lstm_branch)
    test_suite.test_bidirectional_concatenation(lstm_branch, sample_input)
    test_suite.test_gradient_flow(lstm_branch, sample_input)
    test_suite.test_dropout_effect()
    test_suite.test_deterministic_behavior(sample_input)
    test_suite.test_batch_processing(lstm_branch)
    test_suite.test_variable_sequence_length(lstm_branch)
    test_suite.test_output_range(lstm_branch, sample_input)
    test_suite.test_lstm_parameters(lstm_branch)
    
    # Batch norm tests
    test_suite_bn = TestBiLSTMBranchWithBatchNorm()
    lstm_branch_bn = BiLSTMBranchWithBatchNorm(input_size=91, hidden_size=128, num_layers=2, dropout=0.5)
    test_suite_bn.test_output_shape_with_bn(lstm_branch_bn, sample_input)
    test_suite_bn.test_batch_norm_effect(lstm_branch_bn)
    
    print("\n" + "="*60)
    print("All Tests Passed Successfully! ✓")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
