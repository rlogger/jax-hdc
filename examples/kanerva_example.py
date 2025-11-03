"""Kanerva's 'Dollar of Mexico' example.

This example implements the classic HDC demonstration from:
"What We Mean When We Say 'What's the Dollar of Mexico?':
Prototypes and Mapping in Concept Space"
by Pentti Kanerva (2010)

Paper: https://redwood.berkeley.edu/wp-content/uploads/2020/05/kanerva2010what.pdf
Citation: P. Kanerva, "What We Mean When We Say 'What's the Dollar of Mexico?':
          Prototypes and Mapping in Concept Space," 2010 AAAI Fall Symposium Series.

The example shows how to use HDC to represent and query relationships between
countries, capitals, and currencies using high-dimensional vector operations.
"""

import jax
import jax.numpy as jnp
from jax_hdc import MAP
from jax_hdc.functional import cosine_similarity


def main():
    """Run the Kanerva Dollar of Mexico example."""
    print("=" * 70)
    print("Kanerva's 'Dollar of Mexico' Example")
    print("=" * 70)

    # Initialize
    d = 10000  # number of dimensions
    key = jax.random.PRNGKey(42)
    model = MAP.create(dimensions=d)

    print(f"\nUsing {d}-dimensional hypervectors")

    # Create role hypervectors
    print("\nCreating role hypervectors (keys)...")
    keys_key, values_key = jax.random.split(key)
    keys = model.random(keys_key, (3, d))
    country_key, capital_key, currency_key = keys

    print("  - country (role)")
    print("  - capital (role)")
    print("  - currency (role)")

    # Create filler hypervectors for USA
    print("\nCreating filler hypervectors for United States...")
    usa = model.random(values_key, (d,))
    wdc = model.random(jax.random.split(values_key)[0], (d,))
    usd = model.random(jax.random.split(values_key)[1], (d,))

    print("  - USA (country filler)")
    print("  - Washington D.C. (capital filler)")
    print("  - US Dollar (currency filler)")

    # Create filler hypervectors for Mexico
    print("\nCreating filler hypervectors for Mexico...")
    mex = model.random(jax.random.split(values_key)[0], (d,))
    mxc = model.random(jax.random.split(values_key)[1], (d,))
    mxn = model.random(jax.random.split(jax.random.split(values_key)[1])[0], (d,))

    print("  - Mexico (country filler)")
    print("  - Mexico City (capital filler)")
    print("  - Mexican Peso (currency filler)")

    # Create country representations by binding roles with fillers
    print("\nCreating country representations...")
    print("  US = bind(country, USA) + bind(capital, WDC) + bind(currency, USD)")

    us_values = jnp.stack([usa, wdc, usd])
    us_bound = jax.vmap(model.bind)(keys, us_values)
    us = model.bundle(us_bound, axis=0)

    print("  MX = bind(country, MEX) + bind(capital, MXC) + bind(currency, MXN)")

    mx_values = jnp.stack([mex, mxc, mxn])
    mx_bound = jax.vmap(model.bind)(keys, mx_values)
    mx = model.bundle(mx_bound, axis=0)

    # Create mapping from US to Mexico
    print("\nCreating mapping: US → Mexico")
    print("  Mapping = bind(inverse(US), MX)")

    us_inv = model.inverse(us)
    us_to_mx = model.bind(us_inv, mx)

    # Query: What's the dollar of Mexico?
    print("\n" + "=" * 70)
    print("Query: What's the Dollar of Mexico?")
    print("=" * 70)

    print("\nApplying mapping to US Dollar:")
    print("  Result = bind(Mapping, USD)")

    usd_of_mex = model.bind(us_to_mx, usd)

    # Create memory of all known concepts
    memory = jnp.concatenate([
        keys,      # Role vectors
        us_values, # US fillers
        mx_values  # Mexico fillers
    ], axis=0)

    memory_labels = [
        "country (role)",
        "capital (role)",
        "currency (role)",
        "USA",
        "Washington D.C.",
        "US Dollar",
        "Mexico",
        "Mexico City",
        "Mexican Peso"
    ]

    # Find most similar concept
    print("\nComputing similarity with all known concepts:")
    print("-" * 70)

    similarities = jax.vmap(lambda m: cosine_similarity(usd_of_mex, m))(memory)

    # Sort by similarity
    sorted_indices = jnp.argsort(similarities)[::-1]

    for idx in sorted_indices:
        sim = similarities[idx]
        label = memory_labels[idx]
        bar = "█" * int(sim * 50)
        print(f"  {label:20s} | {bar} {sim:.4f}")

    best_match_idx = sorted_indices[0]
    best_match = memory_labels[best_match_idx]
    best_similarity = similarities[best_match_idx]

    print("-" * 70)
    print(f"\nBest match: {best_match} (similarity: {best_similarity:.4f})")

    if best_match == "Mexican Peso":
        print("\nResult: Correct identification of Mexican Peso")
        print("  Currency mapping Mexico -> Peso analogous to USA -> Dollar")
    else:
        print(f"\nResult: Unexpected match '{best_match}' (expected 'Mexican Peso')")

    # Additional queries
    print("\n" + "=" * 70)
    print("Additional Queries")
    print("=" * 70)

    # Query: What's the capital of Mexico?
    print("\nQuery: What's the capital of Mexico?")
    capital_query = model.bind(mx, model.inverse(country_key))
    capital_query = model.bind(capital_query, capital_key)

    capital_sims = jax.vmap(lambda m: cosine_similarity(capital_query, m))(memory)
    capital_match = jnp.argmax(capital_sims)

    print(f"Answer: {memory_labels[capital_match]} (similarity: {capital_sims[capital_match]:.4f})")

    print("\n" + "=" * 70)
    print("Example Complete")
    print("=" * 70)
    print("\nDemonstrates structured knowledge representation and analogical reasoning")
    print("using hyperdimensional vector operations: bind, bundle, and similarity.")


if __name__ == "__main__":
    main()
