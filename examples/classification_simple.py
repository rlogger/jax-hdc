"""Simple classification example with synthetic data.

This example demonstrates how to use JAX-HDC for classification:
1. Generate synthetic data
2. Encode features into hypervectors
3. Train a centroid classifier
4. Evaluate accuracy
"""

import jax
import jax.numpy as jnp
from jax_hdc import MAP
from jax_hdc.embeddings import RandomEncoder
from jax_hdc.models import CentroidClassifier


def generate_synthetic_data(key, n_samples=1000, n_features=20, n_classes=5):
    """Generate synthetic classification dataset.

    Args:
        key: JAX random key
        n_samples: Number of samples
        n_features: Number of features (discrete values 0-9)
        n_classes: Number of classes

    Returns:
        data: Feature matrix of shape (n_samples, n_features)
        labels: Class labels of shape (n_samples,)
    """
    # Generate random features (discrete values 0-9)
    data = jax.random.randint(key, (n_samples, n_features), 0, 10)

    # Generate labels based on first few features (simple pattern)
    # This creates a learnable pattern
    key_labels = jax.random.split(key)[1]
    labels = jax.random.randint(key_labels, (n_samples,), 0, n_classes)

    # Add some structure: samples with similar features get same label
    # Modify labels based on sum of first 3 features
    feature_sum = jnp.sum(data[:, :3], axis=1)
    labels = feature_sum % n_classes

    return data, labels


def main():
    """Run classification example."""
    print("=" * 70)
    print("JAX-HDC: Simple Classification Example")
    print("=" * 70)

    # Configuration
    dimensions = 10000
    n_features = 20
    n_values = 10  # Features are discrete 0-9
    n_classes = 5
    n_train = 800
    n_test = 200

    print(f"\nConfiguration:")
    print(f"  Hypervector dimensions: {dimensions}")
    print(f"  Number of features: {n_features}")
    print(f"  Values per feature: {n_values}")
    print(f"  Number of classes: {n_classes}")
    print(f"  Training samples: {n_train}")
    print(f"  Test samples: {n_test}")

    # Generate data
    print("\n" + "=" * 70)
    print("Step 1: Generate Synthetic Data")
    print("=" * 70)

    key = jax.random.PRNGKey(42)
    data_key, train_key, test_key = jax.random.split(key, 3)

    train_data, train_labels = generate_synthetic_data(
        train_key, n_samples=n_train, n_features=n_features, n_classes=n_classes
    )
    test_data, test_labels = generate_synthetic_data(
        test_key, n_samples=n_test, n_features=n_features, n_classes=n_classes
    )

    print(f"\nGenerated training data: {train_data.shape}")
    print(f"Generated test data: {test_data.shape}")

    # Distribution of classes
    print(f"\nClass distribution (training):")
    for i in range(n_classes):
        count = jnp.sum(train_labels == i)
        print(f"  Class {i}: {count} samples")

    # Create encoder
    print("\n" + "=" * 70)
    print("Step 2: Create Feature Encoder")
    print("=" * 70)

    model = MAP.create(dimensions=dimensions)
    encoder = RandomEncoder.create(
        num_features=n_features,
        num_values=n_values,
        dimensions=dimensions,
        vsa_model=model,
        key=data_key
    )

    print(f"\nCreated RandomEncoder:")
    print(f"  Codebook shape: {encoder.codebook.shape}")
    print(f"  Total parameters: {encoder.codebook.size:,}")

    # Encode data
    print("\n" + "=" * 70)
    print("Step 3: Encode Data into Hypervectors")
    print("=" * 70)

    print("\nEncoding training data...")
    train_hvs = encoder.encode_batch(train_data)
    print(f"Encoded training data shape: {train_hvs.shape}")

    print("\nEncoding test data...")
    test_hvs = encoder.encode_batch(test_data)
    print(f"Encoded test data shape: {test_hvs.shape}")

    # Create and train classifier
    print("\n" + "=" * 70)
    print("Step 4: Train Centroid Classifier")
    print("=" * 70)

    classifier = CentroidClassifier.create(
        num_classes=n_classes,
        dimensions=dimensions,
        vsa_model=model
    )

    print("\nTraining classifier (computing class centroids)...")
    classifier = classifier.fit(train_hvs, train_labels)

    print(f"Trained classifier with {n_classes} prototypes")
    print(f"Prototype shape: {classifier.prototypes.shape}")

    # Evaluate
    print("\n" + "=" * 70)
    print("Step 5: Evaluate Performance")
    print("=" * 70)

    print("\nEvaluating on training data...")
    train_acc = classifier.score(train_hvs, train_labels)
    print(f"Training accuracy: {train_acc:.2%}")

    print("\nEvaluating on test data...")
    test_acc = classifier.score(test_hvs, test_labels)
    print(f"Test accuracy: {test_acc:.2%}")

    # Show predictions for first few test samples
    print("\n" + "=" * 70)
    print("Sample Predictions")
    print("=" * 70)

    n_show = 10
    predictions = classifier.predict(test_hvs[:n_show])
    probs = classifier.predict_proba(test_hvs[:n_show])

    print(f"\nFirst {n_show} test samples:")
    print("-" * 70)
    print(f"{'Sample':<8} {'True':<8} {'Pred':<8} {'Match':<10} {'Confidence':<12}")
    print("-" * 70)

    for i in range(n_show):
        true_label = test_labels[i]
        pred_label = predictions[i]
        correct = "Y" if true_label == pred_label else "N"
        confidence = probs[i, pred_label]

        print(f"{i:<8} {true_label:<8} {pred_label:<8} {correct:<10} {confidence:.4f}")

    print("-" * 70)

    # Confusion analysis
    print("\n" + "=" * 70)
    print("Confusion Matrix")
    print("=" * 70)

    all_predictions = classifier.predict(test_hvs)

    # Compute confusion matrix
    confusion = jnp.zeros((n_classes, n_classes), dtype=jnp.int32)
    for true_label in range(n_classes):
        for pred_label in range(n_classes):
            count = jnp.sum((test_labels == true_label) & (all_predictions == pred_label))
            confusion = confusion.at[true_label, pred_label].set(count)

    print("\n          Predicted")
    print("        " + "  ".join([f"C{i}" for i in range(n_classes)]))
    print("      +" + "-" * (n_classes * 4 + 1))

    for i in range(n_classes):
        row = f"  C{i}  |"
        for j in range(n_classes):
            row += f" {confusion[i, j]:2d} "
        print(row)

    print("\n" + "=" * 70)
    print("Classification Example Complete")
    print("=" * 70)

    print("\nNotes:")
    print("  - Discrete feature encoding via random projection")
    print("  - Centroid classifier: single-pass training")
    print("  - Training time: O(n) in number of samples")
    print("  - Accuracy depends on dimensionality and class separability")


if __name__ == "__main__":
    main()
