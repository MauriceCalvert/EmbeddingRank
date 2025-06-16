import torch
import numpy as np
import math
from typing import Dict, List, Tuple
from embeddingrank import *

def check_entropy_calculation():
    """Test basic entropy calculation against known values."""
    print("=== TESTING ENTROPY CALCULATION ===")

    # Test 1: Uniform distribution should have maximum entropy
    uniform_dist = torch.ones(4) * 0.25  # [0.25, 0.25, 0.25, 0.25]
    expected_entropy = math.log2(4)  # 2.0 bits
    actual_entropy = compute_entropy(uniform_dist * 1000)  # Scale to counts

    print(f"Uniform distribution (4 values):")
    print(f"  Expected entropy: {expected_entropy:.3f} bits")
    print(f"  Actual entropy: {actual_entropy:.3f} bits")
    print(f"  ✓ PASS" if abs(actual_entropy - expected_entropy) < 0.01 else "  ❌ FAIL")

    # Test 2: Deterministic distribution should have zero entropy
    deterministic_dist = torch.tensor([1000.0, 0.0, 0.0, 0.0])
    expected_entropy = 0.0
    actual_entropy = compute_entropy(deterministic_dist)

    print(f"Deterministic distribution:")
    print(f"  Expected entropy: {expected_entropy:.3f} bits")
    print(f"  Actual entropy: {actual_entropy:.3f} bits")
    print(f"  ✓ PASS" if abs(actual_entropy - expected_entropy) < 0.01 else "  ❌ FAIL")
    print()

def check_mutual_information():
    """Test mutual information with known relationships."""
    print("=== TESTING MUTUAL INFORMATION ===")
    device = torch.device('cpu')

    # Test 1: Perfect correlation
    n_samples = 10000
    x = torch.randint(0, 5, (n_samples,), dtype=torch.float32)
    y = x.clone()  # Perfect correlation

    mi = compute_discrete_mutual_information(x, y, device)
    x_entropy = compute_entropy(torch.bincount(x.long()).float())

    print(f"Perfect correlation test:")
    print(f"  X entropy: {x_entropy:.3f} bits")
    print(f"  MI(X,Y): {mi:.3f} bits")
    print(f"  Expected: MI ≈ H(X) for perfect correlation")
    print(f"  ✓ PASS" if abs(mi - x_entropy) < 0.1 else "  ❌ FAIL")

    # Test 2: Independence
    x = torch.randint(0, 5, (n_samples,), dtype=torch.float32)
    y = torch.randint(0, 5, (n_samples,), dtype=torch.float32)

    mi = compute_discrete_mutual_information(x, y, device)

    print(f"Independence test:")
    print(f"  MI(X,Y): {mi:.3f} bits")
    print(f"  Expected: MI ≈ 0 for independent variables")
    print(f"  ✓ PASS" if mi < 0.1 else "  ❌ FAIL")

    # Test 3: Injective deterministic function (corrected)
    x = torch.arange(n_samples, dtype=torch.float32) % 8  # 8 unique values
    y = (x + 3) % 8  # Injective: each x maps to unique y

    mi = compute_discrete_mutual_information(x, y, device)
    x_entropy = compute_entropy(torch.bincount(x.long()).float())
    y_entropy = compute_entropy(torch.bincount(y.long()).float())

    print(f"Injective function test:")
    print(f"  X entropy: {x_entropy:.3f} bits")
    print(f"  Y entropy: {y_entropy:.3f} bits")
    print(f"  MI(X,Y): {mi:.3f} bits")
    print(f"  Expected: MI ≈ H(X) = H(Y) for injective Y=f(X)")
    print(f"  ✓ PASS" if abs(mi - x_entropy) < 0.1 else "  ❌ FAIL")

    # Test 4: Non-injective function (educational)
    x = torch.arange(n_samples, dtype=torch.float32) % 10  # 10 unique values
    y = (x * 2) % 10  # Non-injective: maps to only 5 values (0,2,4,6,8)

    mi = compute_discrete_mutual_information(x, y, device)
    x_entropy = compute_entropy(torch.bincount(x.long()).float())
    y_entropy = compute_entropy(torch.bincount(y.long()).float())

    print(f"Non-injective function test:")
    print(f"  X entropy: {x_entropy:.3f} bits")
    print(f"  Y entropy: {y_entropy:.3f} bits")
    print(f"  MI(X,Y): {mi:.3f} bits")
    print(f"  Expected: MI ≈ H(Y) < H(X) for non-injective Y=f(X)")
    # For non-injective functions, MI(X,Y) = H(Y) ≤ H(X)
    y_mi_match = abs(mi - y_entropy) < 0.1
    mi_le_x_entropy = mi <= x_entropy + 0.1
    print(f"  ✓ PASS" if y_mi_match and mi_le_x_entropy else "  ❌ FAIL")

def check_transition_matrix():
    """Test transition matrix construction with known sequences."""
    print("=== TESTING TRANSITION MATRIX ===")
    device = torch.device('cpu')

    # Test 1: Deterministic sequence 0->1->2->0->1->2...
    sequence = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.float32)
    matrix = build_transition_matrix(sequence, device)

    print(f"Deterministic cycle sequence:")
    print(f"  Matrix shape: {matrix.shape}")
    print(f"  Matrix:\n{matrix}")

    # Check if transitions are deterministic
    expected = torch.zeros(3, 3)
    expected[0, 1] = 1.0  # 0 always goes to 1
    expected[1, 2] = 1.0  # 1 always goes to 2
    expected[2, 0] = 1.0  # 2 always goes to 0

    diff = torch.abs(matrix - expected).max()
    print(f"  Max difference from expected: {diff:.6f}")
    print(f"  ✓ PASS" if diff < 0.01 else "  ❌ FAIL")

    # Test 2: Random sequence should have more uniform transitions
    torch.manual_seed(42)
    random_sequence = torch.randint(0, 3, (1000,), dtype=torch.float32)
    random_matrix = build_transition_matrix(random_sequence, device)

    print(f"Random sequence:")
    print(f"  Matrix shape: {random_matrix.shape}")
    print(f"  Row sums: {random_matrix.sum(dim=1)}")  # Should all be ~1.0

    row_sum_check = torch.allclose(random_matrix.sum(dim=1), torch.ones(3), atol=1e-6)
    print(f"  ✓ PASS" if row_sum_check else "  ❌ FAIL")
    print()

def check_sequential_complexity():
    """Test sequential complexity with known patterns."""
    print("=== TESTING SEQUENTIAL COMPLEXITY ===")
    device = torch.device('cpu')

    # Test 1: Highly predictable sequence
    predictable = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 100, dtype=torch.float32)
    complexity_pred = compute_enhanced_higher_order_complexity(predictable, 4, device)

    print(f"Predictable alternating sequence:")
    print(f"  Complexity: {complexity_pred:.3f}")
    print(f"  Expected: Low complexity for predictable pattern")

    # Test 2: Random sequence
    torch.manual_seed(42)
    random_seq = torch.randint(0, 10, (1000,), dtype=torch.float32)
    complexity_rand = compute_enhanced_higher_order_complexity(random_seq, 4, device)

    print(f"Random sequence:")
    print(f"  Complexity: {complexity_rand:.3f}")
    print(f"  Expected: Higher complexity for random pattern")

    print(f"  ✓ PASS" if complexity_rand > complexity_pred else "  ❌ FAIL")
    print()

def check_dimension_allocation():
    """Test dimension allocation with extreme cases."""
    print("=== TESTING DIMENSION ALLOCATION ===")
    device = torch.device('cpu')

    # Test 1: Equal complexity should give equal allocation
    equal_scores = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32, device=device)
    allocations = allocate_dimensions_final(equal_scores, 20, device)

    print(f"Equal complexity scores:")
    print(f"  Scores: {equal_scores}")
    print(f"  Allocations: {allocations}")
    print(f"  Expected: Roughly equal allocation")

    max_diff = (allocations.max() - allocations.min()).item()
    print(f"  Max difference: {max_diff}")
    print(f"  ✓ PASS" if max_diff <= 1 else "  ❌ FAIL")

    # Test 2: Extreme difference should give skewed allocation
    extreme_scores = torch.tensor([1.0, 0.1, 0.1, 0.1], dtype=torch.float32, device=device)
    allocations_extreme = allocate_dimensions_final(extreme_scores, 20, device)

    print(f"Extreme complexity difference:")
    print(f"  Scores: {extreme_scores}")
    print(f"  Allocations: {allocations_extreme}")
    print(f"  Expected: First attribute gets most dimensions")

    first_gets_most = allocations_extreme[0] > allocations_extreme[1:].max()
    print(f"  ✓ PASS" if first_gets_most else "  ❌ FAIL")

    # Test 3: Total allocation should equal budget
    total_allocated = allocations_extreme.sum().item()
    print(f"  Total allocated: {total_allocated} (should be 20)")
    print(f"  ✓ PASS" if total_allocated == 20 else "  ❌ FAIL")
    print()

def check_known_musical_patterns():
    """Test with synthetic musical data that has known structure."""
    print("=== TESTING WITH KNOWN MUSICAL PATTERNS ===")
    device = torch.device('cpu')

    # Create synthetic musical sequence with known patterns
    n_samples = 5000

    # Pitch: C major scale pattern (0,2,4,5,7,9,11 in 12-tone)
    c_major = torch.tensor([0, 2, 4, 5, 7, 9, 11], dtype=torch.float32)
    pitch = c_major[torch.randint(0, 7, (n_samples,))]

    # Octave: Mostly in middle register (octaves 3-5)
    octave = torch.randint(3, 6, (n_samples,), dtype=torch.float32)

    # Duration: Mostly quarter notes (16/64ths) with some variation
    duration = torch.tensor([16.0] * int(n_samples * 0.7) +
                            [8.0] * int(n_samples * 0.2) +
                            [32.0] * int(n_samples * 0.1))[:n_samples]
    duration = duration[torch.randperm(n_samples)]

    # Delta: Correlated with duration (longer notes = longer gaps)
    delta = duration + torch.randn(n_samples) * 2
    delta = torch.clamp(delta, 0, 64)

    # Timing: Continuous version of delta
    timing = delta / 64.0

    # Semitones: Related to pitch changes
    semitones = torch.diff(pitch, prepend=pitch[0:1])

    # Combine into dataset
    musical_data = torch.stack([pitch, octave, duration, delta, timing, semitones], dim=1)

    # Test the analysis
    attribute_names = ['Pitch', 'Octave', 'Duration', 'Delta', 'Timing', 'Semitones']
    attribute_types = ['discrete', 'discrete', 'discrete', 'discrete', 'continuous', 'continuous']

    allocations, budget = analyze_embedding_requirements(
        data=musical_data,
        attribute_names=attribute_names,
        attribute_types=attribute_types,
        total_embedding_dim=24,
        k_gram_size=4,
        device=device
    )

    print(f"Musical pattern analysis:")
    print(f"  Allocations: {allocations.tolist()}")
    print(f"  Budget recommendations: {budget}")

    # CORRECTED: Validate that correlations were detected in the MI analysis
    # Check if Delta emerged as highly complex due to its engineered correlations
    duration_idx, delta_idx = 2, 3
    duration_alloc = allocations[duration_idx].item()
    delta_alloc = allocations[delta_idx].item()

    print(f"  Duration allocation: {duration_alloc}")
    print(f"  Delta allocation: {delta_alloc}")

    # Test 1: Delta should get more dimensions due to higher complexity
    delta_gets_more = delta_alloc >= duration_alloc
    print(f"  Delta gets ≥ dimensions than Duration: {delta_gets_more}")

    # Test 2: Delta should be identified as most complex (gets most dimensions)
    delta_is_max = delta_alloc == allocations.max().item()
    print(f"  Delta gets most dimensions: {delta_is_max}")

    # Test 3: Total allocation should be exact
    total_correct = allocations.sum().item() == 24
    print(f"  Total allocation correct: {total_correct}")

    # Combined test: engineered correlations were properly detected
    correlations_detected = delta_gets_more and delta_is_max and total_correct
    print(f"  Expected: Engineered correlations properly detected")
    print(f"  ✓ PASS" if correlations_detected else "  ❌ FAIL")

def run_validation_suite():
    """Run all validation tests."""
    print("🧪 EMBEDDINGRANK VALIDATION SUITE")
    print("=" * 50)

    try:
        check_entropy_calculation()
        check_mutual_information()
        check_transition_matrix()
        check_sequential_complexity()
        check_dimension_allocation()
        check_known_musical_patterns()

        print("✅ VALIDATION SUITE COMPLETED")
        print("Review individual test results above.")

    except Exception as e:
        print(f"❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_validation_suite()