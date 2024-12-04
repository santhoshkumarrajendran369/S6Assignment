import torch
import torch.nn as nn
import pytest
from model import Net

def count_parameters(model):
    """Count the total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    """Test if the model has less than 20k parameters"""
    model = Net()
    param_count = count_parameters(model)
    print(f"Total parameters: {param_count}")
    assert param_count < 20000, f"Model has {param_count} parameters, which exceeds 20k limit"

def test_batch_normalization():
    """Test if the model uses batch normalization"""
    model = Net()
    has_batchnorm = any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    assert has_batchnorm, "Model does not use batch normalization"
    
    # Count number of batch norm layers
    batchnorm_count = sum(1 for module in model.modules() if isinstance(module, nn.BatchNorm2d))
    print(f"Number of BatchNorm layers: {batchnorm_count}")

def test_dropout():
    """Test if the model uses dropout"""
    model = Net()
    has_dropout = any(isinstance(module, nn.Dropout) for module in model.modules())
    assert has_dropout, "Model does not use dropout"
    
    # Get dropout values
    dropout_values = [module.p for module in model.modules() if isinstance(module, nn.Dropout)]
    print(f"Dropout values used: {dropout_values}")

def test_gap_no_fc():
    """Test if model uses Global Average Pooling instead of Fully Connected layers"""
    model = Net()
    
    # Check for absence of Linear layers
    has_linear = any(isinstance(module, nn.Linear) for module in model.modules())
    assert not has_linear, "Model should not use fully connected layers"
    
    # Check forward pass shape
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 28, 28)  # MNIST input shape
    output = model(input_tensor)
    
    # Check if output has the correct shape (batch_size, num_classes)
    assert output.shape == (batch_size, 10), f"Expected output shape (1, 10), got {output.shape}"

def test_model_summary():
    """Print model summary for visual inspection"""
    model = Net()
    print("\nModel Architecture:")
    print("==================")
    
    def print_model_structure(model, indent=0):
        for name, child in model.named_children():
            print("  " * indent + f"|- {name}: {child.__class__.__name__}")
            if list(child.children()):
                print_model_structure(child, indent + 1)
    
    print_model_structure(model)

if __name__ == "__main__":
    # Run all tests and print results
    print("Running model architecture tests...\n")
    
    try:
        test_parameter_count()
        print("[PASS] Parameter count test passed")
    except AssertionError as e:
        print(f"[FAIL] Parameter count test failed: {str(e)}")
    
    try:
        test_batch_normalization()
        print("[PASS] Batch normalization test passed")
    except AssertionError as e:
        print(f"[FAIL] Batch normalization test failed: {str(e)}")
    
    try:
        test_dropout()
        print("[PASS] Dropout test passed")
    except AssertionError as e:
        print(f"[FAIL] Dropout test failed: {str(e)}")
    
    try:
        test_gap_no_fc()
        print("[PASS] GAP/FC test passed")
    except AssertionError as e:
        print(f"[FAIL] GAP/FC test failed: {str(e)}")
    
    # Print model summary
    test_model_summary()
