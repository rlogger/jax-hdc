"""Basic operations example for JAX-HDC.

Demonstrates fundamental operations of Hyperdimensional Computing:
- Random hypervector generation
- Binding (dissimilar vector combination)
- Bundling (similar vector aggregation)
- Similarity computation
- Permutation for sequence encoding
"""

import jax
import jax.numpy as jnp
from jax_hdc import MAP, BSC


def demo_binding_and_unbinding():
    """Demonstrate binding and unbinding operations."""
    print("=" * 60)
    print("Binding and Unbinding Demo")
    print("=" * 60)

    # Create MAP model
    model = MAP.create(dimensions=10000)
    key = jax.random.PRNGKey(42)

    # Generate random hypervectors
    x = model.random(key, (10000,))
    y = model.random(key, (10000,))

    print(f"\nGenerated two random hypervectors of dimension {x.shape[0]}")

    # Check initial similarity
    sim_xy = model.similarity(x, y)
    print(f"Similarity between x and y: {sim_xy:.4f}")
    print("(Random vectors should have similarity near 0)")

    # Bind x and y
    bound = model.bind(x, y)
    print(f"\nBound x and y to create a new hypervector")

    # Check similarity with original vectors
    sim_bound_x = model.similarity(bound, x)
    sim_bound_y = model.similarity(bound, y)
    print(f"Similarity between bound and x: {sim_bound_x:.4f}")
    print(f"Similarity between bound and y: {sim_bound_y:.4f}")
    print("(Bound vector should be dissimilar to both inputs)")

    # Unbind using inverse
    y_inv = model.inverse(y)
    unbound = model.bind(bound, y_inv)
    unbound = unbound / jnp.linalg.norm(unbound)  # Normalize

    sim_unbound_x = model.similarity(unbound, x)
    print(f"\nAfter unbinding (binding with inverse of y):")
    print(f"Similarity with original x: {sim_unbound_x:.4f}")
    print("(Should be high, recovering the original x)")


def demo_bundling():
    """Demonstrate bundling operations."""
    print("\n" + "=" * 60)
    print("Bundling Demo")
    print("=" * 60)

    model = MAP.create(dimensions=10000)
    key = jax.random.PRNGKey(42)

    # Generate multiple random hypervectors
    num_vectors = 5
    vectors = model.random(key, (num_vectors, 10000))

    print(f"\nGenerated {num_vectors} random hypervectors")

    # Bundle them together
    bundled = model.bundle(vectors, axis=0)

    print(f"Bundled all vectors together")

    # Check similarity with each input
    print(f"\nSimilarity of bundled vector with each input:")
    for i in range(num_vectors):
        sim = model.similarity(bundled, vectors[i])
        print(f"  Vector {i+1}: {sim:.4f}")

    print("(Bundled vector should be similar to all inputs)")


def demo_sequence_encoding():
    """Demonstrate sequence encoding using permutation."""
    print("\n" + "=" * 60)
    print("Sequence Encoding Demo")
    print("=" * 60)

    from jax_hdc.functional import permute

    model = MAP.create(dimensions=10000)
    key = jax.random.PRNGKey(42)

    # Create symbols for a sequence
    a = model.random(key, (10000,))
    b = model.random(jax.random.split(key)[0], (10000,))
    c = model.random(jax.random.split(key)[1], (10000,))

    print("Created three symbols: A, B, C")

    # Encode sequence [A, B, C] using position-dependent permutation
    # Position 0: permute by 2, Position 1: permute by 1, Position 2: no permutation
    seq_abc = permute(a, 2) + permute(b, 1) + c
    seq_abc = seq_abc / jnp.linalg.norm(seq_abc)

    # Encode different sequence [C, B, A]
    seq_cba = permute(c, 2) + permute(b, 1) + a
    seq_cba = seq_cba / jnp.linalg.norm(seq_cba)

    print("\nEncoded two sequences:")
    print("  Sequence 1: [A, B, C]")
    print("  Sequence 2: [C, B, A]")

    # Check similarity
    sim = model.similarity(seq_abc, seq_cba)
    print(f"\nSimilarity between sequences: {sim:.4f}")
    print("(Different sequences should have low similarity)")

    # Query for symbol at position 0 in seq_abc
    query = permute(seq_abc, -2)  # Unpermute position 0
    sim_a = model.similarity(query, a)
    sim_b = model.similarity(query, b)
    sim_c = model.similarity(query, c)

    print(f"\nQuerying for symbol at position 0 in sequence [A, B, C]:")
    print(f"  Similarity with A: {sim_a:.4f}")
    print(f"  Similarity with B: {sim_b:.4f}")
    print(f"  Similarity with C: {sim_c:.4f}")
    print("(Should match A)")


def demo_bsc_vs_map():
    """Compare BSC and MAP models."""
    print("\n" + "=" * 60)
    print("Comparing BSC vs MAP Models")
    print("=" * 60)

    key = jax.random.PRNGKey(42)

    # BSC model (binary)
    bsc = BSC.create(dimensions=10000)
    x_bsc = bsc.random(key, (10000,))
    y_bsc = bsc.random(key, (10000,))

    print(f"\nBSC (Binary Spatter Codes):")
    print(f"  Vector dtype: {x_bsc.dtype}")
    print(f"  Memory per vector: {x_bsc.nbytes / 1024:.2f} KB")

    bound_bsc = bsc.bind(x_bsc, y_bsc)
    sim_bsc = bsc.similarity(bound_bsc, x_bsc)
    print(f"  Similarity after binding: {sim_bsc:.4f}")

    # MAP model (real-valued)
    map_model = MAP.create(dimensions=10000)
    x_map = map_model.random(key, (10000,))
    y_map = map_model.random(key, (10000,))

    print(f"\nMAP (Multiply-Add-Permute):")
    print(f"  Vector dtype: {x_map.dtype}")
    print(f"  Memory per vector: {x_map.nbytes / 1024:.2f} KB")

    bound_map = map_model.bind(x_map, y_map)
    sim_map = map_model.similarity(bound_map, x_map)
    print(f"  Similarity after binding: {sim_map:.4f}")

    print(f"\nBSC is more memory-efficient ({x_bsc.nbytes / x_map.nbytes:.1f}x smaller)")
    print("MAP provides smooth optimization landscape for gradient-based learning")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("JAX-HDC: Basic Operations Examples")
    print("=" * 60)

    demo_binding_and_unbinding()
    demo_bundling()
    demo_sequence_encoding()
    demo_bsc_vs_map()

    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
