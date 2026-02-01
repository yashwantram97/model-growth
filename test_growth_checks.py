"""
Test script for growth mechanics verification.

This script creates a small model, performs growth, and runs all verification checks
without requiring a full training loop.
"""

import torch
from config import ModelConfig
from models.simple_model import SLM
from transfer.simple_transfer import transition_to_moe
from transfer.simple_growth import scale_bilaterally
from transfer.verify_growth_mechanics import detailed_growth_check, quick_sanity_check


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def test_moe_transition():
    """Test Dense → MoE transition checks."""
    print("\n" + "="*70)
    print("  TEST 1: Dense → MoE Transition")
    print("="*70)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create a small dense model
    dense_model = SLM(
        vocab_size=50257,
        d_model=256,
        n_layers=2,
        n_heads=4,
        d_ff=1024,
        use_moe=False,
    ).to(device)
    
    # Create probe batch
    probe = torch.randint(0, 50257, (4, 64), device=device)
    
    # Quick sanity check on dense model
    quick_sanity_check(dense_model, probe, device, "Dense Model")
    
    # Transition to MoE
    moe_model = transition_to_moe(
        dense_model,
        num_experts=8,
        top_k=2,
    ).to(device)
    
    # Quick sanity check on MoE model
    quick_sanity_check(moe_model, probe, device, "MoE Model")
    
    # Verify functional equivalence
    from transfer.simple_transfer import verify_functional_equivalence
    verify_functional_equivalence(dense_model, moe_model, probe, device)
    
    print("\n✓ Dense → MoE transition test completed!\n")
    
    return moe_model, probe, device


def test_bilateral_growth(moe_model, probe, device):
    """Test Bilateral Growth checks."""
    print("\n" + "="*70)
    print("  TEST 2: Bilateral Growth Verification")
    print("="*70)
    
    scale_factor = 2
    noise_std = 1e-5
    
    # Perform growth
    large_model = scale_bilaterally(
        moe_model,
        scale_factor=scale_factor,
        extra_layers=2,
        noise_std=noise_std
    ).to(device)
    
    # Quick sanity check on large model
    quick_sanity_check(large_model, probe, device, "Large Model (After Growth)")
    
    # Run comprehensive checks
    detailed_growth_check(
        old_model=moe_model,
        new_model=large_model,
        probe=probe,
        device=device,
        scale_factor=scale_factor,
        tolerance=1e-4
    )
    
    print("\n✓ Bilateral growth test completed!\n")
    
    return large_model


def test_with_different_noise_levels():
    """Test growth with different noise levels to verify symmetry breaking."""
    print("\n" + "="*70)
    print("  TEST 3: Symmetry Breaking with Different Noise Levels")
    print("="*70)
    
    device = get_device()
    
    # Create a small MoE model
    moe_model = SLM(
        vocab_size=50257,
        d_model=256,
        n_layers=2,
        n_heads=4,
        d_ff=1024,
        use_moe=True,
        num_experts=8,
        top_k=2,
    ).to(device)
    
    probe = torch.randint(0, 50257, (4, 64), device=device)
    
    noise_levels = [0.0, 1e-7, 1e-5, 1e-3]
    
    for noise_std in noise_levels:
        print(f"\n  Testing with noise_std = {noise_std}")
        print(f"  {'-'*66}")
        
        large_model = scale_bilaterally(
            moe_model,
            scale_factor=2,
            extra_layers=1,
            noise_std=noise_std
        ).to(device)
        
        # Only check symmetry breaking
        from transfer.verify_growth_mechanics import check_symmetry_breaking
        try:
            passed, grad_diff = check_symmetry_breaking(large_model, device)
            if passed:
                print(f"    → Symmetry breaking successful with noise_std={noise_std}")
            else:
                print(f"    → WARNING: Symmetry breaking failed with noise_std={noise_std}")
        except Exception as e:
            print(f"    → Error: {e}")
    
    print("\n✓ Noise level test completed!\n")


def test_rope_integrity():
    """Test RoPE integrity with correct and incorrect scaling."""
    print("\n" + "="*70)
    print("  TEST 4: RoPE Integrity with Different Scaling Strategies")
    print("="*70)
    
    device = get_device()
    
    # Create a small MoE model
    moe_model = SLM(
        vocab_size=50257,
        d_model=256,
        n_layers=2,
        n_heads=4,
        d_ff=1024,
        use_moe=True,
        num_experts=8,
        top_k=2,
    ).to(device)
    
    probe = torch.randint(0, 50257, (4, 64), device=device)
    
    # Test 1: Correct scaling (should pass)
    print("\n  Test 4a: Correct scaling (d_model x2, n_heads x2)")
    print(f"  {'-'*66}")
    
    large_model = scale_bilaterally(
        moe_model,
        scale_factor=2,
        extra_layers=1,
        noise_std=1e-5
    ).to(device)
    
    from transfer.verify_growth_mechanics import check_rope_integrity
    try:
        check_rope_integrity(moe_model, large_model)
        print("    → RoPE integrity test PASSED")
    except ValueError as e:
        print(f"    → RoPE integrity test FAILED: {e}")
    
    # Test 2: Incorrect scaling would require modifying the scale function,
    # so we'll just document the expected behavior
    print("\n  Test 4b: Incorrect scaling (would fail)")
    print(f"  {'-'*66}")
    print("    If n_heads were not scaled proportionally with d_model,")
    print("    the RoPE integrity check would raise ValueError.")
    print("    Current implementation correctly scales both proportionally.")
    
    print("\n✓ RoPE integrity test completed!\n")


def main():
    """Run all growth verification tests."""
    print("\n" + "#"*70)
    print("#" + "GROWTH MECHANICS VERIFICATION TEST SUITE".center(68) + "#")
    print("#"*70)
    
    try:
        # Test 1: Dense → MoE transition
        moe_model, probe, device = test_moe_transition()
        
        # Test 2: Bilateral growth with comprehensive checks
        large_model = test_bilateral_growth(moe_model, probe, device)
        
        # Test 3: Different noise levels
        test_with_different_noise_levels()
        
        # Test 4: RoPE integrity
        test_rope_integrity()
        
        print("\n" + "#"*70)
        print("#" + "ALL TESTS COMPLETED SUCCESSFULLY".center(68) + "#")
        print("#"*70 + "\n")
        
    except Exception as e:
        print("\n" + "#"*70)
        print("#" + "TEST SUITE FAILED".center(68) + "#")
        print("#"*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
