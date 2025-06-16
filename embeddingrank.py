"""
EmbeddingRank: Information-Theoretic Dimension Allocation for Multivariate Sequence Embeddings

A framework for analyzing multivariate time series data and automatically determining
optimal embedding dimension allocation based on information content, sequential complexity, and
cross-attribute dependencies. This approach goes beyond simple concatenation or summation by using
principled information-theoretic measures to allocate embedding dimensions proportionally to the
complexity of each attribute.

Key Features:
- Information content analysis using entropy measures
- Sequential complexity detection including k-gram analysis and medium-range patterns
- Cross-attribute dependency analysis using mutual information
- Adaptive dimension allocation with budget recommendations
- Support for mixed discrete/continuous data

Author: Maurice Calvert, maurice AT calvert DOT ch
License: MIT
"""

import math
import torch
import numpy as np
from typing import Dict, List, Tuple, Union
from collections import defaultdict

def allocate_dimensions_final(
        complexity_scores: torch.Tensor,
        total_dim: int,
        device: torch.device
) -> torch.Tensor:
    """
    Allocate embedding dimensions based on complexity scores with guaranteed minimum allocation.

    Uses a two-stage allocation strategy: first ensures minimum dimensions per attribute,
    then distributes remaining dimensions proportionally to complexity scores.

    Args:
        complexity_scores: Normalized complexity scores for each attribute [N]
        total_dim: Total embedding dimensions to allocate
        device: Torch device for computations

    Returns:
        Tensor of integer dimension allocations for each attribute [N]

    Raises:
        AssertionError: If complexity_scores is empty or total_dim is too small
    """
    assert len(complexity_scores) > 0, "Complexity scores cannot be empty"
    assert total_dim >= len(
        complexity_scores), f"Total dimensions {total_dim} must be at least {len(complexity_scores)}"

    min_dims_per_attr: int = max(1, total_dim // (len(complexity_scores) * 4))
    available_dims: int = total_dim - len(complexity_scores) * min_dims_per_attr

    if torch.sum(complexity_scores) > 0:
        proportions: torch.Tensor = complexity_scores / torch.sum(complexity_scores)
        additional_dims: torch.Tensor = torch.round(proportions * available_dims)
    else:
        additional_dims: torch.Tensor = torch.zeros_like(complexity_scores)

    allocations: torch.Tensor = torch.full_like(complexity_scores, min_dims_per_attr, dtype=torch.int32, device=device)
    allocations = allocations + additional_dims.int()

    # Handle rounding errors by adjusting highest complexity attributes first
    diff: int = total_dim - torch.sum(allocations).item()
    if diff != 0:
        sorted_indices: torch.Tensor = torch.argsort(complexity_scores, descending=True)
        i: int
        for i in range(abs(diff)):
            idx: int = sorted_indices[i % len(sorted_indices)].item()
            if diff > 0:
                allocations[idx] += 1
            else:
                allocations[idx] = max(1, allocations[idx] - 1)

    return allocations

def analyze_embedding_requirements(
        data: torch.Tensor,
        attribute_names: List[str],
        attribute_types: List[str],
        total_embedding_dim: int = None,
        k_gram_size: int = 8,
        device: torch.device = None
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Main entry point for analyzing embedding requirements of multivariate time series data.

    Performs comprehensive analysis including information content, sequential complexity,
    and cross-attribute dependencies to determine optimal embedding dimension allocation.

    Args:
        data: Input tensor of shape [L, N] where L=samples, N=attributes
        attribute_names: Human-readable names for each attribute
        attribute_types: Type specification for each attribute ('discrete' or 'continuous')
        total_embedding_dim: Fixed budget constraint, None for auto-recommendation
        k_gram_size: Sequential pattern analysis depth for discrete attributes
        device: Computing device, auto-detects if None

    Returns:
        Tuple containing:
            - allocations: Tensor of embedding dimensions per attribute [N]
            - budget_recommendations: Dict with 'minimum', 'recommended', 'optimal' budgets

    Raises:
        AssertionError: If data dimensions don't match attribute specifications
        ValueError: If invalid attribute types are provided

    Example:
        >>> data = torch.randn(10000, 4)  # 10k samples, 4 attributes
        >>> names = ['temperature', 'humidity', 'wind_speed', 'pressure']
        >>> types = ['continuous', 'continuous', 'continuous', 'continuous']
        >>> allocations, budgets = analyze_embedding_requirements(data, names, types)
        >>> print(f"Recommended budget: {budgets['recommended']} dimensions")
        >>> print(f"Allocations: {allocations.tolist()}")
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = data.to(device)

    allocations: torch.Tensor
    budget_recommendations: Dict[str, int]
    allocations, budget_recommendations = compute_embedding_allocations(
        data=data,
        attribute_types=attribute_types,
        attribute_names=attribute_names,
        total_embedding_dim=total_embedding_dim,
        k_gram_size=k_gram_size,
        device=device
    )

    return allocations, budget_recommendations

def build_transition_matrix(
        sequence: torch.Tensor,
        device: torch.device
) -> torch.Tensor:
    """
    Build a first-order Markov transition matrix from a discrete sequence.

    Constructs a stochastic matrix where entry (i,j) represents the probability
    of transitioning from state i to state j in the next time step.

    Args:
        sequence: Discrete sequence values [L]
        device: Torch device for computations

    Returns:
        Row-normalized transition matrix [vocab_size, vocab_size]

    Note:
        Sequence values are automatically shifted to start from 0 for indexing.
        Empty rows (states that never occur) are given uniform distributions.
        Vocabulary size is computed from actual data range.
    """
    if len(sequence) < 2:
        return torch.eye(1, device=device)

    # Move to CPU to avoid CUDA indexing issues
    sequence_cpu = sequence.cpu()
    min_val: int = int(sequence_cpu.min().item())
    max_val: int = int(sequence_cpu.max().item())
    normalized_seq: torch.Tensor = (sequence_cpu - min_val).long()

    # Compute vocabulary size from actual data range
    vocab_size: int = max_val - min_val + 1

    curr_states: torch.Tensor = normalized_seq[:-1]
    next_states: torch.Tensor = normalized_seq[1:]

    indices: torch.Tensor = torch.stack([curr_states, next_states], dim=0)
    values: torch.Tensor = torch.ones(len(curr_states), dtype=torch.float32)

    # Fix: correct argument order for sparse_coo_tensor
    transition_counts: torch.Tensor = torch.sparse_coo_tensor(
        indices,
        values,
        (vocab_size, vocab_size),
        dtype=torch.float32
    ).to_dense()

    row_sums: torch.Tensor = transition_counts.sum(dim=1, keepdim=True)
    row_sums = torch.clamp(row_sums, min=1e-8)
    normalized_matrix = transition_counts / row_sums

    # Move back to target device
    return normalized_matrix.to(device)

def compute_autoregressive_complexity(
        data: torch.Tensor,
        max_order: int = 5
) -> float:
    """
    Estimate sequential complexity of continuous time series using autoregressive modeling.

    Fits AR models of increasing order and measures prediction complexity based on
    model order and residual variance. Higher complexity indicates stronger temporal
    dependencies requiring more embedding capacity.

    Args:
        data: Continuous time series values [L]
        max_order: Maximum AR model order to test

    Returns:
        Complexity measure combining model order and prediction error (capped at 10.0)

    Note:
        Uses normalized data and linear least squares for AR coefficient estimation.
        Returns 1.0 for insufficient data or numerical issues.
    """
    data_np: np.ndarray = data.cpu().numpy()

    if len(data_np) < 100:
        return 1.0

    # Normalize data to prevent numerical issues
    data_norm: np.ndarray = (data_np - np.mean(data_np)) / (np.std(data_np) + 1e-8)
    best_complexity: float = 1.0

    order: int
    for order in range(1, min(max_order + 1, len(data_norm) // 10)):
        try:
            # Create lagged feature matrix
            X: np.ndarray = np.column_stack([
                data_norm[i:len(data_norm) - order + i] for i in range(order)
            ])
            y: np.ndarray = data_norm[order:]

            if len(X) < 50:
                continue

            # Fit AR model using normal equations
            XtX: np.ndarray = X.T @ X + 1e-6 * np.eye(order)
            Xty: np.ndarray = X.T @ y
            coeffs: np.ndarray = np.linalg.solve(XtX, Xty)

            # Compute prediction error
            y_pred: np.ndarray = X @ coeffs
            residual_var: float = np.var(y - y_pred)

            # Complexity increases with model order and decreases with prediction accuracy
            complexity: float = order * (1.0 + 1.0 / (residual_var + 1e-6))
            best_complexity = max(best_complexity, complexity)

        except Exception:
            continue

    return min(best_complexity, 10.0)

def compute_continuous_mutual_information(
        x: torch.Tensor,
        y: torch.Tensor
) -> float:
    """
    Estimate mutual information between continuous variables using correlation proxy.

    Converts Pearson correlation to approximate mutual information using the
    Gaussian assumption: MI ≈ -0.5 * log(1 - ρ²).

    Args:
        x: First continuous variable [N]
        y: Second continuous variable [N]

    Returns:
        Estimated mutual information in bits (non-negative)

    Note:
        This is an approximation valid for Gaussian variables. For non-Gaussian
        distributions, considers using kernel density estimation methods.
    """
    try:
        corr: float = torch.corrcoef(torch.stack([x, y]))[0, 1].abs().item()
        # Convert correlation to approximate MI using Gaussian assumption
        return max(0.0, -0.5 * math.log(1 - corr ** 2 + 1e-10))
    except Exception:
        return 0.0

def compute_differential_entropy(
        data: torch.Tensor,
        n_bins: int = 50
) -> float:
    """
    Estimate differential entropy of continuous data using histogram method.

    Discretizes the data using quantile-based binning and estimates the
    differential entropy by correcting the discrete entropy for bin width.

    Args:
        data: Continuous values to analyze [N]
        n_bins: Number of histogram bins for discretization

    Returns:
        Estimated differential entropy in nats (≥ 1.0)

    Algorithm:
        1. Remove extreme outliers (1st/99th percentiles)
        2. Create uniform-width histogram
        3. Compute discrete entropy: H_discrete = -Σ p_i * log(p_i)
        4. Add bin width correction: H_differential = H_discrete + log(bin_width)

    Note:
        Returns 1.0 fallback for insufficient data or numerical issues.
    """
    data_np: np.ndarray = data.cpu().numpy()

    # Remove outliers for robust binning
    q1: float
    q99: float
    q1, q99 = np.percentile(data_np, [1, 99])
    data_clean: np.ndarray = data_np[(data_np >= q1) & (data_np <= q99)]

    if len(data_clean) < 100:
        return 1.0

    # Compute histogram with uniform bin width
    counts: np.ndarray
    bin_edges: np.ndarray
    counts, bin_edges = np.histogram(data_clean, bins=n_bins)
    bin_width: float = bin_edges[1] - bin_edges[0]

    # Convert to probability distribution
    probs: np.ndarray = counts / len(data_clean)
    probs = probs[probs > 0]  # Remove zero probabilities

    # Differential entropy estimate
    discrete_entropy: float = -np.sum(probs * np.log(probs))
    differential_entropy: float = discrete_entropy + np.log(bin_width)

    return max(1.0, differential_entropy)

def compute_discrete_mutual_information(
        x: torch.Tensor,
        y: torch.Tensor,
        device: torch.device
) -> float:
    # Move to CPU for numerical stability
    x_cpu = x.cpu()
    y_cpu = y.cpu()

    # Get unique values and check cardinality
    x_vals = torch.unique(x_cpu, sorted=True)
    y_vals = torch.unique(y_cpu, sorted=True)

    x_card = len(x_vals)
    y_card = len(y_vals)

    # For extremely high cardinality, use stratified sampling
    if x_card > 10000 or y_card > 10000:
        n_samples = len(x_cpu)
        if n_samples > 50000:
            indices = torch.randperm(n_samples)[:50000]
            x_cpu = x_cpu[indices]
            y_cpu = y_cpu[indices]
            x_vals = torch.unique(x_cpu, sorted=True)
            y_vals = torch.unique(y_cpu, sorted=True)
            x_card = len(x_vals)
            y_card = len(y_vals)

    # If still too high cardinality, use correlation proxy
    if x_card > 10000 or y_card > 10000:
        try:
            x_float = x_cpu.float()
            y_float = y_cpu.float()
            corr = torch.corrcoef(torch.stack([x_float, y_float]))[0, 1].abs()
            return max(0.0, -0.5 * math.log2(1 - corr.item() ** 2 + 1e-10))
        except:
            return 0.0

    # Use searchsorted for efficient O(n log k) mapping instead of loops
    x_map = torch.searchsorted(x_vals, x_cpu)
    y_map = torch.searchsorted(y_vals, y_cpu)

    # Build joint histogram
    joint_indices = x_map * y_card + y_map
    joint_hist = torch.bincount(joint_indices, minlength=x_card * y_card)
    joint_hist = joint_hist.view(x_card, y_card).float()

    # Convert to probabilities
    joint_probs = joint_hist / len(x_cpu)
    x_marginal = joint_probs.sum(dim=1)
    y_marginal = joint_probs.sum(dim=0)

    # Vectorized MI computation with numerical stability
    mask = joint_probs > 1e-12
    if not torch.any(mask):
        return 0.0

    joint_nonzero = joint_probs[mask]
    x_indices, y_indices = torch.where(mask)
    x_marg_selected = x_marginal[x_indices]
    y_marg_selected = y_marginal[y_indices]

    independence_probs = torch.clamp(x_marg_selected * y_marg_selected, min=1e-12)
    log_ratios = torch.log2(joint_nonzero / independence_probs)
    mi_terms = joint_nonzero * log_ratios

    return max(0.0, torch.sum(mi_terms).item())
def compute_entropy_from_counts(counts: torch.Tensor) -> float:
    """
    Compute Shannon entropy from count tensor.

    Args:
        counts: Count tensor [V] where V is vocabulary size

    Returns:
        Shannon entropy in bits
    """
    # Remove zero counts
    counts = counts[counts > 0].float()
    if len(counts) == 0:
        return 0.0

    # Convert to probabilities
    probs = counts / torch.sum(counts)

    # Compute entropy: H(X) = -Σ p(x) * log2(p(x))
    log_probs = torch.log2(probs)
    entropy = -torch.sum(probs * log_probs).item()

    return entropy

def validate_mi_properties(x: torch.Tensor, y: torch.Tensor, device: torch.device) -> Dict[str, float]:
    """
    Validate that MI computation satisfies mathematical properties.

    Returns:
        Dictionary with validation metrics
    """
    mi_xy = compute_discrete_mutual_information(x, y, device)
    mi_yx = compute_discrete_mutual_information(y, x, device)  # Should be symmetric

    # Compute individual entropies
    x_counts = torch.bincount((x - x.min()).long())
    y_counts = torch.bincount((y - y.min()).long())

    h_x = compute_entropy_from_counts(x_counts.float())
    h_y = compute_entropy_from_counts(y_counts.float())

    return {
        'MI(X,Y)': mi_xy,
        'MI(Y,X)': mi_yx,
        'H(X)': h_x,
        'H(Y)': h_y,
        'symmetry_error': abs(mi_xy - mi_yx),
        'mi_le_hx': mi_xy <= h_x + 1e-6,  # MI(X,Y) ≤ H(X)
        'mi_le_hy': mi_xy <= h_y + 1e-6,  # MI(X,Y) ≤ H(Y)
    }

# Test the fixed implementation
def test_fixed_mi():
    """Test the fixed MI implementation with the previously failing cases."""
    device = torch.device('cpu')

    print("=== TESTING FIXED MUTUAL INFORMATION ===")

    # Test 1: Perfect correlation (should have MI = H(X))
    n_samples = 10000
    x = torch.randint(0, 5, (n_samples,), dtype=torch.float32)
    y = x.clone()

    validation = validate_mi_properties(x, y, device)
    print(f"Perfect correlation:")
    for key, value in validation.items():
        print(f"  {key}: {value}")

    mi_hx_ratio = validation['MI(X,Y)'] / validation['H(X)']
    print(f"  MI/H(X) ratio: {mi_hx_ratio:.4f} (should be ≈ 1.0)")
    print(f"  ✓ PASS" if abs(mi_hx_ratio - 1.0) < 0.05 else "  ❌ FAIL")
    print()

    # Test 2: Deterministic function Y = f(X)
    x = torch.arange(n_samples, dtype=torch.float32) % 10
    y = (x * 2) % 10  # Deterministic function

    validation = validate_mi_properties(x, y, device)
    print(f"Deterministic function Y = (X*2) % 10:")
    for key, value in validation.items():
        print(f"  {key}: {value}")

    # For deterministic Y=f(X), MI should equal H(X) if f is injective within the domain
    mi_hx_ratio = validation['MI(X,Y)'] / validation['H(X)']
    print(f"  MI/H(X) ratio: {mi_hx_ratio:.4f}")
    print(f"  ✓ PASS" if abs(mi_hx_ratio - 1.0) < 0.1 else "  ❌ FAIL")
    print()

    # Test 3: High cardinality (like the Delta case)
    x = torch.arange(5000, dtype=torch.float32)
    y = x + torch.randn(5000) * 10  # Noisy relationship

    validation = validate_mi_properties(x, y, device)
    print(f"High cardinality with noise:")
    for key, value in validation.items():
        print(f"  {key}: {value}")

    # Should detect some relationship (MI > 0) but not perfect correlation
    mi_value = validation['MI(X,Y)']
    print(f"  MI > 0: {mi_value > 0}")
    print(f"  MI reasonable: {0.1 < mi_value < validation['H(X)']}")
    print(f"  ✓ PASS" if mi_value > 0.1 and mi_value < validation['H(X)'] else "  ❌ FAIL")
    print()

if __name__ == "__main__":
    test_fixed_mi()
def compute_embedding_allocations(
        data: torch.Tensor,
        attribute_types: List[str],
        attribute_names: List[str] = None,
        total_embedding_dim: int = None,
        k_gram_size: int = 8,
        device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Core algorithm for computing embedding dimension allocations based on comprehensive complexity analysis.

    Performs a six-step analysis pipeline:
    1. Information content analysis (entropy measures)
    2. Sequential complexity analysis (temporal patterns)
    3. Cross-attribute dependency analysis (mutual information)
    4. Unified complexity scoring (weighted combination)
    5. Budget recommendations (information-theoretic bounds)
    6. Dimension allocation (proportional to complexity)

    Args:
        data: Multivariate time series tensor [L, N] where L=samples, N=attributes
        attribute_types: Type specification for each attribute ('discrete' or 'continuous')
        attribute_names: Optional human-readable names, defaults to Attr_0, Attr_1, ...
        total_embedding_dim: Optional budget constraint, uses recommended if None
        k_gram_size: Sequential pattern analysis depth for discrete sequences
        device: Computing device for tensor operations

    Returns:
        Tuple containing:
            - allocations: Integer dimension allocation per attribute [N]
            - budget_recommendations: Dict with budget analysis

    Raises:
        AssertionError: If input dimensions don't match or parameters are invalid

    Complexity:
        Time: O(L*N + N²) for MI computation, O(L*K) for k-gram analysis
        Space: O(V²) where V is maximum vocabulary size
    """
    assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D"
    assert len(attribute_types) == data.shape[
        1], f"Attribute types length {len(attribute_types)} must match data columns {data.shape[1]}"

    data = data.to(device)
    L: int = data.shape[0]
    N: int = data.shape[1]

    assert L > 0 and N > 0, f"Invalid tensor shape: {data.shape}"
    assert k_gram_size >= 2, f"k-gram size must be at least 2"

    print(f"=== EMBEDDING ALLOCATION ANALYSIS ===")
    print(f"Device: {device}")
    print(f"Dataset: {L:,} samples, {N} attributes")
    print(f"Attribute types: {attribute_types}")
    print(f"Analysis depth: {k_gram_size}-grams for discrete, medium-range patterns")
    print()

    # Initialize attribute names if not provided
    if attribute_names is None:
        attribute_names = [f"Attr_{i}" for i in range(N)]
    else:
        assert len(attribute_names) == N, f"Attribute names length {len(attribute_names)} must match data columns {N}"

    # Step 1: Information content analysis
    information_contents: torch.Tensor = torch.zeros(N, dtype=torch.float32, device=device)
    print("STEP 1: INFORMATION CONTENT ANALYSIS")
    print("Measuring theoretical minimum dimensions for each attribute type.")

    i: int
    for i in range(N):
        print(f"  {attribute_names[i]} ({attribute_types[i]}):")

        if attribute_types[i] == 'discrete':
            unique_vals: torch.Tensor = torch.unique(data[:, i])
            vocab_size: int = len(unique_vals)

            freq_dist: torch.Tensor = torch.bincount((data[:, i] - data[:, i].min()).long()).to(device)
            entropy: float = compute_entropy(freq_dist.float())
            information_contents[i] = entropy

            print(f"    Vocabulary size: {vocab_size}")
            print(f"    Entropy: {entropy:.3f} bits")
            print(f"    Theoretical min dimensions: {math.ceil(entropy)}")

        elif attribute_types[i] == 'continuous':
            # Estimate differential entropy using quantization
            diff_entropy: float = compute_differential_entropy(data[:, i])
            effective_entropy: float = max(1.0, diff_entropy)
            information_contents[i] = effective_entropy

            print(f"    Value range: [{data[:, i].min().item():.3f}, {data[:, i].max().item():.3f}]")
            print(f"    Differential entropy: {diff_entropy:.3f} nats")
            print(f"    Effective dimensions: {math.ceil(effective_entropy)}")

        else:
            raise ValueError(f"Unknown attribute type: {attribute_types[i]}. Must be 'discrete' or 'continuous'")

        # Interpretation
        complexity_level: str = interpret_information_content(information_contents[i].item(), attribute_types[i])
        print(f"    → INTERPRETATION: {complexity_level}")
    print()

    # Step 2: Sequential complexity analysis
    print("STEP 2: SEQUENTIAL COMPLEXITY ANALYSIS")
    print("Measuring temporal/sequential patterns including medium-range dependencies.")
    sequential_complexities: torch.Tensor = torch.zeros(N, dtype=torch.float32, device=device)

    for i in range(N):
        print(f"  {attribute_names[i]} ({attribute_types[i]}):")

        if attribute_types[i] == 'discrete':
            # Short-range Markov analysis
            transition_matrix: torch.Tensor = build_transition_matrix(data[:, i], device)
            markov_complexity: float = compute_matrix_complexity(transition_matrix)

            # Enhanced k-gram analysis
            kgram_complexity: float
            if L > k_gram_size * 50:
                kgram_complexity = compute_enhanced_higher_order_complexity(
                    data[:, i], k_gram_size, device
                )
                print(f"    {k_gram_size}-gram complexity: {kgram_complexity:.3f}")
            else:
                kgram_complexity = markov_complexity

            # Medium-range pattern detection
            pattern_complexity: float = detect_medium_range_patterns(data[:, i], device)
            print(f"    Medium-range pattern complexity: {pattern_complexity:.3f}")

            # Combined complexity
            sequential_complexities[i] = max(markov_complexity, kgram_complexity, pattern_complexity)
            print(f"    Final sequential complexity: {sequential_complexities[i]:.3f}")

            # Interpretation using absolute thresholds
            complexity_value: float = sequential_complexities[i].item()
            if complexity_value > 5.0:
                print(f"    → INTERPRETATION: Highly random transitions, complex patterns")
            elif complexity_value < 2.0:
                print(f"    → INTERPRETATION: Strong sequential patterns, predictable")
            else:
                print(f"    → INTERPRETATION: Moderate sequential structure")

        elif attribute_types[i] == 'continuous':
            # Use autoregressive analysis for continuous sequences
            ar_complexity: float = compute_autoregressive_complexity(data[:, i])
            sequential_complexities[i] = ar_complexity
            print(f"    Autoregressive complexity: {ar_complexity:.3f}")

            if ar_complexity > 5.0:
                print(f"    → INTERPRETATION: High temporal complexity, strong memory effects")
            elif ar_complexity > 2.0:
                print(f"    → INTERPRETATION: Moderate temporal dependencies")
            else:
                print(f"    → INTERPRETATION: Weak temporal structure")
    print()

    # Step 3: Cross-attribute dependencies
    print("STEP 3: CROSS-ATTRIBUTE DEPENDENCY ANALYSIS")
    print("Measuring information sharing between attributes (discrete/continuous/mixed).")
    dependency_scores: torch.Tensor = torch.zeros(N, dtype=torch.float32, device=device)

    for i in range(N):
        cross_deps: List[float] = []
        print(f"  {attribute_names[i]} dependencies:")

        j: int
        for j in range(N):
            if i != j:
                mi: float = compute_mixed_mutual_information(
                    data[:, i], data[:, j], attribute_types[i], attribute_types[j], device
                )
                cross_deps.append(mi)
                print(f"    I({attribute_names[i]},{attribute_names[j]}): {mi:.4f}")

                # Interpretation
                dep_strength: str = interpret_dependency_strength(mi)
                print(f"      → {dep_strength}")

        dependency_scores[i] = sum(cross_deps)
        print(f"    Total dependency score: {dependency_scores[i]:.4f}")

        # Overall interpretation
        if dependency_scores[i] > 4.0:
            print(f"    → INTERPRETATION: Highly connected, needs substantial coordination capacity")
        elif dependency_scores[i] > 1.5:
            print(f"    → INTERPRETATION: Moderately connected, some shared structure")
        else:
            print(f"    → INTERPRETATION: Mostly independent")
    print()

    # Step 4: Unified complexity scoring
    print("STEP 4: UNIFIED COMPLEXITY SCORING")
    print("Combining information content, enhanced sequential patterns, and dependencies.")

    # Normalize all metrics to [0,1] range
    info_norm: torch.Tensor = information_contents / torch.max(information_contents)
    seq_norm: torch.Tensor = (sequential_complexities / torch.max(sequential_complexities)
                              if torch.max(sequential_complexities) > 0 else sequential_complexities)
    dep_norm: torch.Tensor = (dependency_scores / torch.max(dependency_scores)
                              if torch.max(dependency_scores) > 0 else dependency_scores)

    # Weighted combination emphasizing sequential patterns
    alpha: float = 0.35  # Information content weight
    beta: float = 0.4  # Sequential complexity weight (emphasized)
    gamma: float = 0.25  # Dependency weight

    combined_scores: torch.Tensor = alpha * info_norm + beta * seq_norm + gamma * dep_norm

    print("  Normalized complexity components:")
    for i in range(N):
        print(f"    {attribute_names[i]}:")
        print(f"      Information (normalized): {info_norm[i]:.3f}")
        print(f"      Sequential (normalized): {seq_norm[i]:.3f}")
        print(f"      Dependencies (normalized): {dep_norm[i]:.3f}")
        print(f"      Combined score: {combined_scores[i]:.3f}")

        if combined_scores[i] > 0.8:
            print(f"      → OVERALL: HIGH COMPLEXITY - needs substantial embedding capacity")
        elif combined_scores[i] > 0.5:
            print(f"      → OVERALL: MODERATE COMPLEXITY - standard allocation")
        else:
            print(f"      → OVERALL: LOW COMPLEXITY - minimal embedding sufficient")
    print()

    # Step 5: Budget recommendations
    print("STEP 5: EMBEDDING BUDGET RECOMMENDATIONS")

    # Information-theoretic minimum
    min_budget: int = sum(math.ceil(information_contents[i].item()) for i in range(N))

    # Adaptive complexity multipliers
    avg_complexity: float = torch.mean(combined_scores).item()
    complexity_variance: float = torch.var(combined_scores).item()

    # Enhanced multiplier accounting for sequential patterns
    complexity_multiplier: float = 1.2 + 1.5 * avg_complexity + complexity_variance
    safety_factor: float = 1.25  # 25% margin for medium-range patterns

    recommended_budget: int = max(min_budget, int(min_budget * complexity_multiplier * safety_factor))

    # Optimal budget with diminishing returns threshold
    optimal_multiplier: float = 2.5 + 2.0 * complexity_variance
    optimal_budget: int = int(min_budget * optimal_multiplier)

    budget_recommendations: Dict[str, int] = {
        'minimum': min_budget,
        'recommended': recommended_budget,
        'optimal': optimal_budget
    }

    print(f"  Minimum viable budget: {min_budget} dimensions")
    print(f"    → Information-theoretic floor, expect information loss below this")
    print(f"  Recommended budget: {recommended_budget} dimensions")
    print(f"    → Enhanced complexity-adjusted for medium-range patterns")
    print(f"  Optimal budget: {optimal_budget} dimensions")
    print(f"    → Diminishing returns threshold, maximum practical benefit")
    print()

    # Use recommended budget if none provided
    if total_embedding_dim is None:
        total_embedding_dim = recommended_budget
        print(f"Using recommended budget: {total_embedding_dim} dimensions")
    else:
        print(f"User-specified budget: {total_embedding_dim} dimensions")

        if total_embedding_dim < min_budget:
            print(f"  ⚠️  WARNING: Below minimum viable budget, expect significant information loss")
        elif total_embedding_dim < recommended_budget:
            print(f"  ⚠️  CAUTION: Below recommended budget, may miss medium-range patterns")
        elif total_embedding_dim > optimal_budget:
            print(f"  ℹ️  INFO: Above optimal budget, diminishing returns expected")
        else:
            print(f"  ✓ GOOD: Within recommended range for enhanced pattern detection")
    print()

    # Step 6: Dimension allocation
    print("STEP 6: DIMENSION ALLOCATION")
    allocations: torch.Tensor = allocate_dimensions_final(combined_scores, total_embedding_dim, device)

    print("  Final allocations:")
    total_allocated: int = 0
    baseline_equal: int = total_embedding_dim // N

    for i in range(N):
        allocation: int = allocations[i].item()
        percentage: float = (allocation / total_embedding_dim) * 100
        vs_equal: int = allocation - baseline_equal

        print(f"    {attribute_names[i]} ({attribute_types[i]}): {allocation} dims ({percentage:.1f}%)")

        if vs_equal > 3:
            print(f"      → +{vs_equal} dims vs equal split - ABOVE AVERAGE complexity")
        elif vs_equal < -3:
            print(f"      → {vs_equal} dims vs equal split - BELOW AVERAGE complexity")
        else:
            print(f"      → {vs_equal:+d} dims vs equal split - AVERAGE complexity")

        total_allocated += allocation

    print(f"  Total allocated: {total_allocated} dims")
    print(f"  Allocation efficiency: {(total_allocated / total_embedding_dim) * 100:.1f}%")

    # Summary analysis
    print()
    print("=== SUMMARY ===")
    max_score_idx: int = torch.argmax(combined_scores).item()
    min_score_idx: int = torch.argmin(combined_scores).item()

    print(f"Most complex attribute: {attribute_names[max_score_idx]} (gets {allocations[max_score_idx].item()} dims)")
    print(f"Least complex attribute: {attribute_names[min_score_idx]} (gets {allocations[min_score_idx].item()} dims)")

    score_spread: float = (torch.max(combined_scores) - torch.min(combined_scores)).item()
    if score_spread > 0.5:
        print("Recommendation: HETEROGENEOUS allocation strongly justified")
    elif score_spread > 0.2:
        print("Recommendation: MODERATE allocation differences justified")
    else:
        print("Recommendation: EQUAL allocation sufficient - similar complexities")

    return allocations, budget_recommendations

def compute_enhanced_higher_order_complexity(
        sequence: torch.Tensor,
        k: int,
        device: torch.device
) -> float:
    """
    Compute higher-order sequential complexity using enhanced k-gram entropy analysis.

    Analyzes conditional entropy patterns in k-grams to detect complex sequential
    dependencies beyond first-order Markov chains. Uses efficient sampling and
    vectorized operations for scalability.

    Args:
        sequence: Discrete sequence values [L]
        k: Order of n-gram analysis (context length = k-1)
        vocab_size: Number of unique values in vocabulary
        device: Torch device for computations

    Returns:
        Average conditional entropy across valid contexts (1.0 to vocab_size*0.8)

    Algorithm:
        1. Sample sequence efficiently if too long
        2. Extract all k-grams and their contexts
        3. Group by context and compute conditional entropies
        4. Return average entropy across contexts with sufficient samples

    Note:
        Returns 0.0 for insufficient data (< k*50 samples).
        Uses vocabulary capping to prevent integer overflow in large vocabularies.
    """
    if len(sequence) < k * 50:
        return 0.0
    vocab_size = len(torch.unique(sequence))
    # Efficient sampling strategy preserving temporal structure
    max_samples: int = min(len(sequence), 20000)
    sample_seq: torch.Tensor
    if len(sequence) > max_samples:
        step: int = max(1, len(sequence) // max_samples)
        indices: torch.Tensor = torch.arange(0, len(sequence) - k + 1, step, device=device)[:max_samples - k + 1]
        sample_start: int = indices[0].item()
        sample_seq = sequence[sample_start:sample_start + min(max_samples, len(sequence) - sample_start)]
    else:
        sample_seq = sequence

    # Normalize sequence to start from 0
    context_size: int = k - 1
    min_val: int = sample_seq.min().item()
    normalized_seq: torch.Tensor = sample_seq - min_val

    if len(normalized_seq) < k:
        return 0.0

    # Extract contexts and next values efficiently
    num_grams: int = len(normalized_seq) - k + 1
    contexts_tensor: torch.Tensor = torch.zeros(num_grams, context_size, dtype=torch.int64, device=device)

    i: int
    for i in range(context_size):
        contexts_tensor[:, i] = normalized_seq[i:i + num_grams]

    next_vals: torch.Tensor = normalized_seq[context_size:context_size + num_grams]

    # Convert contexts to unique identifiers for efficient grouping
    context_ids: torch.Tensor = torch.zeros(num_grams, dtype=torch.int64, device=device)
    multiplier: int = 1
    vocab_cap: int = min(vocab_size, 50)  # Prevent overflow

    for i in range(context_size - 1, -1, -1):
        if multiplier > 1e15:  # Numerical stability check
            break
        context_ids += contexts_tensor[:, i] * multiplier
        multiplier *= vocab_cap

    # Group by context and compute conditional entropies
    unique_contexts: torch.Tensor
    inverse_indices: torch.Tensor
    unique_contexts, inverse_indices = torch.unique(context_ids, return_inverse=True)

    total_entropy: float = 0.0
    valid_contexts: int = 0

    ctx_idx: int
    for ctx_idx in range(len(unique_contexts)):
        mask: torch.Tensor = inverse_indices == ctx_idx
        next_vals_for_context: torch.Tensor = next_vals[mask]

        if len(next_vals_for_context) >= 5:  # Minimum samples for reliable entropy
            counts: torch.Tensor = torch.bincount(next_vals_for_context.long(), minlength=vocab_size).float()
            counts = counts[counts > 0]
            if len(counts) > 1:
                entropy: float = compute_entropy(counts)
                total_entropy += entropy
                valid_contexts += 1

    if valid_contexts == 0:
        return 1.0

    avg_entropy: float = total_entropy / valid_contexts
    return max(1.0, min(avg_entropy, float(vocab_size) * 0.8))

def compute_entropy(freq_dist: torch.Tensor) -> float:
    """
    Compute Shannon entropy from frequency distribution.

    Calculates H(X) = -Σ p(x) * log₂(p(x)) where p(x) are normalized probabilities.

    Args:
        freq_dist: Frequency counts for each value [V]

    Returns:
        Shannon entropy in bits (non-negative)

    Note:
        Automatically filters out zero frequencies to avoid log(0).
        Uses base-2 logarithm for entropy in bits.
    """
    freq_dist = freq_dist[freq_dist > 0]
    probs: torch.Tensor = freq_dist / torch.sum(freq_dist)
    entropy: torch.Tensor = -torch.sum(probs * torch.log2(probs))
    return entropy.item()

def compute_matrix_complexity(matrix: torch.Tensor) -> float:
    """
    Compute matrix complexity using effective rank based on singular value decomposition.

    Measures the intrinsic dimensionality of a matrix by counting singular values
    needed to capture 95% of the variance. Higher effective rank indicates more
    complex transition patterns.

    Args:
        matrix: Square matrix to analyze [N, N]

    Returns:
        Effective rank as complexity measure (1.0 to N)

    Algorithm:
        1. Perform SVD: A = UΣV^T
        2. Compute cumulative variance ratios
        3. Count singular values needed for 95% variance

    Note:
        Adds small diagonal regularization for numerical stability.
        Moves to CPU to avoid CUDA device-side assert issues.
    """
    # Move to CPU to avoid CUDA device-side assert issues
    matrix_cpu = matrix.cpu()

    try:
        U: torch.Tensor
        S: torch.Tensor
        V: torch.Tensor
        U, S, V = torch.svd(matrix_cpu + 1e-8 * torch.eye(matrix_cpu.shape[0]))

        total_variance: torch.Tensor = torch.sum(S)
        if total_variance.item() < 1e-10:
            return 1.0

        cumulative_variance: torch.Tensor = torch.cumsum(S, dim=0)
        variance_ratios: torch.Tensor = cumulative_variance / total_variance
        effective_rank: int = torch.sum(variance_ratios < 0.95).item() + 1
        return float(effective_rank)

    except Exception as e:
        print(f"SVD failed: {e}")
        return 1.0

def compute_mixed_mutual_information(
        x: torch.Tensor,
        y: torch.Tensor,
        x_type: str,
        y_type: str,
        device: torch.device
) -> float:
    """
    Compute mutual information between variables of mixed types (discrete/continuous).

    Handles all combinations of discrete and continuous variables using appropriate
    methods for each case. For mixed cases, discretizes continuous variables.

    Args:
        x: First variable [N]
        y: Second variable [N]
        x_type: Type of x ('discrete' or 'continuous')
        y_type: Type of y ('discrete' or 'continuous')
        device: Torch device for computations

    Returns:
        Mutual information estimate in bits (non-negative)

    Cases:
        - Both discrete: Exact MI using joint histograms
        - Both continuous: Correlation-based approximation
        - Mixed: Discretize continuous variable, then compute discrete MI

    Note:
        Uses sampling for efficiency with large datasets (>50k samples).
    """
    # Sample for computational efficiency
    max_samples: int = 50000
    if len(x) > max_samples:
        indices: torch.Tensor = torch.randperm(len(x), device=device)[:max_samples]
        x = x[indices]
        y = y[indices]

    if x_type == 'discrete' and y_type == 'discrete':
        return compute_discrete_mutual_information(x, y, device)
    elif x_type == 'continuous' and y_type == 'continuous':
        return compute_continuous_mutual_information(x, y)
    else:
        # Mixed case: discretize the continuous variable
        if x_type == 'continuous':
            x_discrete: torch.Tensor = discretize_continuous(x, n_bins=20)
            return compute_discrete_mutual_information(x_discrete, y, device)
        else:
            y_discrete: torch.Tensor = discretize_continuous(y, n_bins=20)
            return compute_discrete_mutual_information(x, y_discrete, device)

def count_pattern_repeats_with_gaps(
        sequence: torch.Tensor,
        pattern_len: int,
        max_gap: int,
        device: torch.device
) -> int:
    """
    Count repeated patterns allowing gaps between occurrences.

    Searches for exact pattern matches within a specified gap distance,
    useful for detecting medium-range periodicities and recurring motifs.

    Args:
        sequence: Input sequence to analyze [L]
        pattern_len: Length of patterns to search for
        max_gap: Maximum allowed gap between pattern occurrences
        device: Torch device for computations

    Returns:
        Count of pattern repetitions found

    Algorithm:
        1. Extract all possible patterns of given length
        2. For each pattern, search for matches within max_gap distance
        3. Count total number of repetitions found

    Note:
        Uses sampling for efficiency when sequence has >1000 possible patterns.
        Only counts first repetition found for each unique pattern.
    """
    if len(sequence) < pattern_len * 2:
        return 0

    repeat_count: int = 0

    # Limit search space for efficiency
    num_patterns: int = len(sequence) - pattern_len + 1
    pattern_starts: range
    if num_patterns > 1000:
        step: int = num_patterns // 1000
        pattern_starts = range(0, num_patterns, step)
    else:
        pattern_starts = range(num_patterns)

    start_idx: int
    for start_idx in pattern_starts:
        if start_idx + pattern_len >= len(sequence):
            break

        pattern: torch.Tensor = sequence[start_idx:start_idx + pattern_len]

        # Search for repetitions within allowed gap
        search_start: int = start_idx + pattern_len
        search_end: int = min(len(sequence) - pattern_len + 1, start_idx + max_gap + pattern_len)

        search_idx: int
        for search_idx in range(search_start, search_end):
            if search_idx + pattern_len > len(sequence):
                break

            candidate: torch.Tensor = sequence[search_idx:search_idx + pattern_len]

            # Check for exact match
            if torch.equal(pattern, candidate):
                repeat_count += 1
                break  # Only count first repetition for this pattern

    return repeat_count

def demo_with_synthetic_data() -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Demonstration function showing EmbeddingRank usage with synthetic multivariate data.

    Creates a realistic test dataset with mixed discrete/continuous attributes,
    known correlations, and temporal patterns to showcase the analysis capabilities.

    Returns:
        Tuple containing:
            - allocations: Computed dimension allocations [N]
            - budget_recommendations: Budget analysis results

    Data Generation:
        - 50k samples with 6 attributes (3 discrete, 3 continuous)
        - Introduces correlations between attributes
        - Adds temporal dependencies for demonstration

    Example Usage:
        >>> allocations, budgets = demo_with_synthetic_data()
        >>> print(f"Recommended total dimensions: {budgets['recommended']}")
        >>> print(f"Per-attribute allocation: {allocations.tolist()}")
    """
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    # Generate synthetic multivariate time series
    L: int = 50000
    attributes: List[torch.Tensor] = []
    vocab_sizes: List[int] = [100, 50, 200]

    # Create mixed discrete/continuous attributes
    i: int
    for i in range(6):
        if i % 2 == 0:  # discrete attributes
            vocab_size: int = vocab_sizes[i // 2]
            attr: torch.Tensor = torch.randint(1, vocab_size + 1, (L,), dtype=torch.float32, device=device)
        else:  # continuous attributes
            if i == 1:
                attr = torch.randn(L, device=device) * 2 + 5
            elif i == 3:
                attr = torch.distributions.Exponential(1.0).sample((L,)).to(device)
            else:
                attr = torch.rand(L, device=device) * 10
        attributes.append(attr)

    # Introduce correlations and dependencies
    mask: torch.Tensor = torch.rand(L, device=device) < 0.4
    attributes[2][mask] = (attributes[0][mask] % 50 + 1).float()  # Discrete correlation
    attributes[3] = 0.6 * attributes[1] + 0.4 * torch.randn(L, device=device)  # Continuous correlation

    test_data: torch.Tensor = torch.stack(attributes, dim=1)

    # Define data schema
    attribute_names: List[str] = ['Color', 'Weight', 'Temperature', 'Brightness', 'Size', 'Texture']
    attribute_types: List[str] = ['discrete', 'continuous', 'discrete', 'continuous', 'discrete', 'continuous']

    # Analyze embedding requirements
    allocations: torch.Tensor
    budget_recommendations: Dict[str, int]
    allocations, budget_recommendations = analyze_embedding_requirements(
        data=test_data,
        attribute_names=attribute_names,
        attribute_types=attribute_types,
        total_embedding_dim=None,  # Use recommended budget
        k_gram_size=8
    )

    print(f"\nFinal budget recommendations: {budget_recommendations}")
    return allocations, budget_recommendations

def detect_medium_range_patterns(
        sequence: torch.Tensor,
        device: torch.device,
        max_gap: int = 50
) -> float:
    """
    Detect repeated patterns with medium-range gaps to capture complex temporal dependencies.

    Identifies recurring subsequences that repeat with allowable gaps, indicating
    structured but non-contiguous patterns that require additional embedding capacity.

    Args:
        sequence: Discrete sequence to analyze [L]
        vocab_size: Number of unique values in vocabulary
        device: Torch device for computations
        max_gap: Maximum gap allowed between pattern repetitions

    Returns:
        Pattern complexity score (1.0 to 8.0) based on repetition frequency

    Algorithm:
        1. Sample sequence for efficiency if too long
        2. Search for patterns of lengths [3, 5, 8]
        3. Count repetitions allowing gaps up to max_gap
        4. Combine repetition scores with logarithmic scaling

    Note:
        Returns 1.0 baseline if sequence is too short for analysis.
        Uses logarithmic scaling to prevent explosion from frequent patterns.
    """
    if len(sequence) < max_gap * 2:
        return 1.0

    pattern_complexity: float = 1.0

    # Sample sequence for computational efficiency
    max_samples: int = min(len(sequence), 10000)
    sample_seq: torch.Tensor
    if len(sequence) > max_samples:
        indices: torch.Tensor = torch.randperm(len(sequence), device=device)[:max_samples]
        indices = torch.sort(indices)[0]  # Preserve temporal order
        sample_seq = sequence[indices]
    else:
        sample_seq = sequence

    # Search for patterns of different lengths
    pattern_len: int
    for pattern_len in [3, 5, 8]:
        if len(sample_seq) < pattern_len * 4:  # Need sufficient occurrences
            continue

        pattern_repeats: int = count_pattern_repeats_with_gaps(sample_seq, pattern_len, max_gap, device)

        if pattern_repeats > pattern_len:  # Significant repetition detected
            repeat_factor: float = math.log(pattern_repeats / pattern_len + 1)
            pattern_complexity += repeat_factor

    return min(pattern_complexity, 8.0)

def discretize_continuous(
        data: torch.Tensor,
        n_bins: int = 20
) -> torch.Tensor:
    """
    Convert continuous data to discrete bins using quantile-based binning.

    Uses quantile-based binning for robust discretization that handles outliers
    and skewed distributions better than uniform binning.

    Args:
        data: Continuous values to discretize [N]
        n_bins: Number of discrete bins to create

    Returns:
        Discretized values as integer indices [N]

    Algorithm:
        1. Compute quantile-based bin edges
        2. Remove duplicate edges for robustness
        3. Digitize values using numpy's digitize function

    Note:
        Returns zeros for degenerate cases (all values identical).
        Bin indices start from 0 and go up to n_bins-1.
    """
    data_np: np.ndarray = data.cpu().numpy()
    quantiles: np.ndarray = np.linspace(0, 1, n_bins + 1)
    bin_edges: np.ndarray = np.quantile(data_np, quantiles)
    bin_edges = np.unique(bin_edges)  # Remove duplicates

    if len(bin_edges) < 2:
        return torch.zeros_like(data, dtype=torch.long)

    digitized: np.ndarray = np.digitize(data_np, bin_edges[1:-1])
    return torch.tensor(digitized, dtype=torch.long, device=data.device)

def interpret_dependency_strength(mi: float) -> str:
    """
    Interpret mutual information values with human-readable descriptions.

    Converts MI values to qualitative strength indicators for better understanding
    of attribute relationships in the analysis output.

    Args:
        mi: Mutual information value in bits

    Returns:
        Human-readable strength description

    Thresholds:
        - > 2.0: Strong dependency (highly predictive)
        - > 0.5: Moderate dependency (some correlation)
        - > 0.1: Weak dependency (slight correlation)
        - ≤ 0.1: Independence (no apparent relationship)
    """
    if mi > 2.0:
        return "Strong dependency - highly predictive"
    elif mi > 0.5:
        return "Moderate dependency - some correlation"
    elif mi > 0.1:
        return "Weak dependency - slight correlation"
    else:
        return "Independence - no apparent relationship"

def interpret_information_content(content: float, attr_type: str) -> str:
    """
    Interpret information content values with context-specific descriptions.

    Provides qualitative interpretation of entropy/information content based on
    attribute type, helping users understand the complexity implications.

    Args:
        content: Information content value (entropy in bits or nats)
        attr_type: Attribute type ('discrete' or 'continuous')

    Returns:
        Human-readable complexity description

    Discrete Thresholds:
        - > 10 bits: High entropy (diverse vocabulary)
        - > 5 bits: Moderate entropy (balanced distribution)
        - ≤ 5 bits: Low entropy (skewed distribution)

    Continuous Thresholds:
        - > 3 nats: High variability (complex distribution)
        - > 1.5 nats: Moderate variability (structured distribution)
        - ≤ 1.5 nats: Low variability (concentrated distribution)
    """
    if attr_type == 'discrete':
        if content > 10:
            return "High entropy - very diverse vocabulary"
        elif content > 5:
            return "Moderate entropy - balanced distribution"
        else:
            return "Low entropy - skewed distribution"
    else:  # continuous
        if content > 3:
            return "High variability - complex distribution"
        elif content > 1.5:
            return "Moderate variability - structured distribution"
        else:
            return "Low variability - concentrated distribution"

def main() -> None:
    """
    Main execution function demonstrating EmbeddingRank capabilities.

    Runs the synthetic data demonstration to showcase the complete analysis
    pipeline and dimension allocation process.
    """
    demo_with_synthetic_data()

if __name__ == "__main__":
    main()