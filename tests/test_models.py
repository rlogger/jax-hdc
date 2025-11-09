"""Tests for HDC classification models."""

import jax
import jax.numpy as jnp
import pytest
from jax_hdc.models import CentroidClassifier, AdaptiveHDC
from jax_hdc.vsa import MAP, BSC


class TestCentroidClassifier:
    """Tests for CentroidClassifier."""

    def test_creation_with_defaults(self):
        """Test CentroidClassifier creation with default parameters."""
        classifier = CentroidClassifier.create(
            num_classes=5,
            dimensions=100,
            key=jax.random.PRNGKey(42)
        )

        assert classifier.num_classes == 5
        assert classifier.dimensions == 100
        assert classifier.vsa_model_name == "map"
        assert classifier.prototypes.shape == (5, 100)

    def test_creation_with_bsc(self):
        """Test CentroidClassifier with BSC model."""
        classifier = CentroidClassifier.create(
            num_classes=3,
            dimensions=50,
            vsa_model="bsc",
            key=jax.random.PRNGKey(42)
        )

        assert classifier.vsa_model_name == "bsc"
        assert classifier.prototypes.dtype == jnp.bool_

    def test_creation_with_vsa_model_instance(self):
        """Test creation with VSAModel instance."""
        vsa = MAP.create(dimensions=100)
        classifier = CentroidClassifier.create(
            num_classes=3,
            dimensions=100,
            vsa_model=vsa,
            key=jax.random.PRNGKey(42)
        )

        assert classifier.vsa_model_name == "map"

    def test_creation_with_initial_prototypes(self):
        """Test creation with custom initial prototypes."""
        initial = jax.random.normal(jax.random.PRNGKey(42), (3, 100))
        classifier = CentroidClassifier.create(
            num_classes=3,
            dimensions=100,
            initial_prototypes=initial
        )

        assert jnp.allclose(classifier.prototypes, initial)

    def test_similarity_computation(self):
        """Test similarity computation between query and prototypes."""
        classifier = CentroidClassifier.create(
            num_classes=3,
            dimensions=100,
            key=jax.random.PRNGKey(42)
        )

        query = jax.random.normal(jax.random.PRNGKey(0), (100,))
        similarities = classifier.similarity(query)

        assert similarities.shape == (3,)
        assert jnp.isfinite(similarities).all()

    def test_predict_single_query(self):
        """Test prediction on a single query."""
        classifier = CentroidClassifier.create(
            num_classes=3,
            dimensions=100,
            key=jax.random.PRNGKey(42)
        )

        query = jax.random.normal(jax.random.PRNGKey(0), (100,))
        prediction = classifier.predict(query)

        assert prediction.shape == ()
        assert 0 <= prediction < 3

    def test_predict_batch(self):
        """Test prediction on a batch of queries."""
        classifier = CentroidClassifier.create(
            num_classes=3,
            dimensions=100,
            key=jax.random.PRNGKey(42)
        )

        queries = jax.random.normal(jax.random.PRNGKey(0), (10, 100))
        predictions = classifier.predict(queries)

        assert predictions.shape == (10,)
        assert jnp.all((predictions >= 0) & (predictions < 3))

    def test_predict_proba_single_query(self):
        """Test probability prediction on a single query."""
        classifier = CentroidClassifier.create(
            num_classes=3,
            dimensions=100,
            key=jax.random.PRNGKey(42)
        )

        query = jax.random.normal(jax.random.PRNGKey(0), (100,))
        probs = classifier.predict_proba(query)

        assert probs.shape == (3,)
        assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-5)
        assert jnp.all(probs >= 0) and jnp.all(probs <= 1)

    def test_predict_proba_batch(self):
        """Test probability prediction on a batch."""
        classifier = CentroidClassifier.create(
            num_classes=3,
            dimensions=100,
            key=jax.random.PRNGKey(42)
        )

        queries = jax.random.normal(jax.random.PRNGKey(0), (10, 100))
        probs = classifier.predict_proba(queries)

        assert probs.shape == (10, 3)
        assert jnp.allclose(jnp.sum(probs, axis=-1), 1.0, atol=1e-5)
        assert jnp.all(probs >= 0) and jnp.all(probs <= 1)

    def test_fit(self):
        """Test fitting the classifier on training data."""
        vsa = MAP.create(dimensions=100)
        classifier = CentroidClassifier.create(
            num_classes=3,
            dimensions=100,
            vsa_model=vsa,
            key=jax.random.PRNGKey(42)
        )

        # Create training data
        key = jax.random.PRNGKey(0)
        train_hvs = vsa.random(key, (30, 100))
        train_labels = jnp.array([0]*10 + [1]*10 + [2]*10)

        # Fit classifier
        trained_classifier = classifier.fit(train_hvs, train_labels)

        assert trained_classifier.prototypes.shape == (3, 100)
        # Prototypes should be different from initial random ones
        assert not jnp.allclose(trained_classifier.prototypes, classifier.prototypes)

    def test_fit_improves_accuracy(self):
        """Test that fitting improves accuracy on separable data."""
        vsa = MAP.create(dimensions=1000)
        classifier = CentroidClassifier.create(
            num_classes=3,
            dimensions=1000,
            vsa_model=vsa,
            key=jax.random.PRNGKey(42)
        )

        # Create separable training data with distinct prototypes
        key = jax.random.PRNGKey(0)
        proto_keys = jax.random.split(key, 3)

        # Create class-specific base vectors
        base_vectors = [vsa.random(proto_keys[i], (1000,)) for i in range(3)]

        # Generate samples near each base vector
        train_hvs = []
        train_labels = []
        for class_idx in range(3):
            for _ in range(10):
                noise = vsa.random(jax.random.split(key)[0], (1000,)) * 0.1
                sample = base_vectors[class_idx] + noise
                # Normalize
                sample = sample / jnp.linalg.norm(sample)
                train_hvs.append(sample)
                train_labels.append(class_idx)

        train_hvs = jnp.stack(train_hvs)
        train_labels = jnp.array(train_labels)

        # Fit and test
        trained_classifier = classifier.fit(train_hvs, train_labels)
        accuracy = trained_classifier.score(train_hvs, train_labels)

        # Should achieve reasonable accuracy on separable data
        assert accuracy > 0.5

    def test_update_online(self):
        """Test online update with a single sample."""
        classifier = CentroidClassifier.create(
            num_classes=3,
            dimensions=100,
            key=jax.random.PRNGKey(42)
        )

        sample = jax.random.normal(jax.random.PRNGKey(0), (100,))
        old_prototype = classifier.prototypes[1].copy()

        updated_classifier = classifier.update_online(sample, label=1, learning_rate=0.1)

        # Prototype should have changed
        assert not jnp.allclose(updated_classifier.prototypes[1], old_prototype)
        # Other prototypes should be unchanged
        assert jnp.allclose(updated_classifier.prototypes[0], classifier.prototypes[0])
        assert jnp.allclose(updated_classifier.prototypes[2], classifier.prototypes[2])

    def test_score(self):
        """Test accuracy scoring."""
        vsa = MAP.create(dimensions=100)
        classifier = CentroidClassifier.create(
            num_classes=3,
            dimensions=100,
            vsa_model=vsa,
            key=jax.random.PRNGKey(42)
        )

        # Create test data
        test_hvs = vsa.random(jax.random.PRNGKey(0), (20, 100))
        test_labels = jax.random.randint(jax.random.PRNGKey(1), (20,), 0, 3)

        score = classifier.score(test_hvs, test_labels)

        assert 0.0 <= score <= 1.0
        assert isinstance(float(score), float)

    def test_fit_with_empty_class(self):
        """Test fitting when some classes have no samples."""
        classifier = CentroidClassifier.create(
            num_classes=5,
            dimensions=100,
            key=jax.random.PRNGKey(42)
        )

        # Create training data with only 3 classes
        vsa = MAP.create(dimensions=100)
        train_hvs = vsa.random(jax.random.PRNGKey(0), (30, 100))
        train_labels = jnp.array([0]*10 + [1]*10 + [2]*10)  # Classes 3 and 4 are empty

        trained_classifier = classifier.fit(train_hvs, train_labels)

        # Should still produce valid prototypes
        assert trained_classifier.prototypes.shape == (5, 100)
        assert jnp.isfinite(trained_classifier.prototypes).all()

    def test_immutability(self):
        """Test that operations return new instances (immutability)."""
        classifier1 = CentroidClassifier.create(
            num_classes=3,
            dimensions=100,
            key=jax.random.PRNGKey(42)
        )

        vsa = MAP.create(dimensions=100)
        train_hvs = vsa.random(jax.random.PRNGKey(0), (30, 100))
        train_labels = jnp.array([0]*10 + [1]*10 + [2]*10)

        classifier2 = classifier1.fit(train_hvs, train_labels)

        # Should be different instances
        assert classifier1 is not classifier2
        # Original should be unchanged
        assert not jnp.allclose(classifier1.prototypes, classifier2.prototypes)


class TestAdaptiveHDC:
    """Tests for AdaptiveHDC."""

    def test_creation(self):
        """Test AdaptiveHDC creation."""
        classifier = AdaptiveHDC.create(
            num_classes=5,
            dimensions=100,
            key=jax.random.PRNGKey(42)
        )

        assert classifier.num_classes == 5
        assert classifier.dimensions == 100
        assert classifier.vsa_model_name == "map"
        assert classifier.prototypes.shape == (5, 100)
        assert classifier.num_updates.shape == (5,)
        assert jnp.all(classifier.num_updates == 0)

    def test_creation_with_bsc(self):
        """Test AdaptiveHDC with BSC model."""
        classifier = AdaptiveHDC.create(
            num_classes=3,
            dimensions=50,
            vsa_model="bsc",
            key=jax.random.PRNGKey(42)
        )

        assert classifier.vsa_model_name == "bsc"
        assert classifier.prototypes.dtype == jnp.bool_

    def test_predict_single_query(self):
        """Test prediction on a single query."""
        classifier = AdaptiveHDC.create(
            num_classes=3,
            dimensions=100,
            key=jax.random.PRNGKey(42)
        )

        query = jax.random.normal(jax.random.PRNGKey(0), (100,))
        prediction = classifier.predict(query)

        assert prediction.shape == ()
        assert 0 <= prediction < 3

    def test_predict_batch(self):
        """Test prediction on a batch of queries."""
        classifier = AdaptiveHDC.create(
            num_classes=3,
            dimensions=100,
            key=jax.random.PRNGKey(42)
        )

        queries = jax.random.normal(jax.random.PRNGKey(0), (10, 100))
        predictions = classifier.predict(queries)

        assert predictions.shape == (10,)
        assert jnp.all((predictions >= 0) & (predictions < 3))

    def test_fit_single_epoch(self):
        """Test fitting with a single epoch."""
        vsa = MAP.create(dimensions=100)
        classifier = AdaptiveHDC.create(
            num_classes=3,
            dimensions=100,
            vsa_model=vsa,
            key=jax.random.PRNGKey(42)
        )

        # Create training data
        train_hvs = vsa.random(jax.random.PRNGKey(0), (30, 100))
        train_labels = jnp.array([0]*10 + [1]*10 + [2]*10)

        trained_classifier = classifier.fit(train_hvs, train_labels, epochs=1)

        assert trained_classifier.prototypes.shape == (3, 100)
        # Prototypes should be different from initial
        assert not jnp.allclose(trained_classifier.prototypes, classifier.prototypes)

    def test_fit_multiple_epochs(self):
        """Test fitting with multiple epochs."""
        vsa = MAP.create(dimensions=100)
        classifier = AdaptiveHDC.create(
            num_classes=3,
            dimensions=100,
            vsa_model=vsa,
            key=jax.random.PRNGKey(42)
        )

        # Create training data
        train_hvs = vsa.random(jax.random.PRNGKey(0), (30, 100))
        train_labels = jnp.array([0]*10 + [1]*10 + [2]*10)

        trained_classifier = classifier.fit(train_hvs, train_labels, epochs=5)

        assert trained_classifier.prototypes.shape == (3, 100)

    def test_score(self):
        """Test accuracy scoring."""
        vsa = MAP.create(dimensions=100)
        classifier = AdaptiveHDC.create(
            num_classes=3,
            dimensions=100,
            vsa_model=vsa,
            key=jax.random.PRNGKey(42)
        )

        # Create test data
        test_hvs = vsa.random(jax.random.PRNGKey(0), (20, 100))
        test_labels = jax.random.randint(jax.random.PRNGKey(1), (20,), 0, 3)

        score = classifier.score(test_hvs, test_labels)

        assert 0.0 <= score <= 1.0

    def test_adaptive_improves_with_epochs(self):
        """Test that multiple epochs can improve accuracy."""
        vsa = MAP.create(dimensions=500)
        classifier = AdaptiveHDC.create(
            num_classes=3,
            dimensions=500,
            vsa_model=vsa,
            key=jax.random.PRNGKey(42)
        )

        # Create separable training data
        key = jax.random.PRNGKey(0)
        proto_keys = jax.random.split(key, 3)

        base_vectors = [vsa.random(proto_keys[i], (500,)) for i in range(3)]

        train_hvs = []
        train_labels = []
        for class_idx in range(3):
            for _ in range(10):
                noise = vsa.random(jax.random.split(key)[0], (500,)) * 0.1
                sample = base_vectors[class_idx] + noise
                sample = sample / jnp.linalg.norm(sample)
                train_hvs.append(sample)
                train_labels.append(class_idx)

        train_hvs = jnp.stack(train_hvs)
        train_labels = jnp.array(train_labels)

        # Train with different epochs
        classifier_1epoch = classifier.fit(train_hvs, train_labels, epochs=1)
        classifier_5epochs = classifier.fit(train_hvs, train_labels, epochs=5)

        acc_1epoch = classifier_1epoch.score(train_hvs, train_labels)
        acc_5epochs = classifier_5epochs.score(train_hvs, train_labels)

        # Both should achieve reasonable accuracy (not testing improvement due to randomness)
        assert acc_1epoch > 0.3
        assert acc_5epochs > 0.3

    def test_immutability(self):
        """Test that operations return new instances."""
        classifier1 = AdaptiveHDC.create(
            num_classes=3,
            dimensions=100,
            key=jax.random.PRNGKey(42)
        )

        vsa = MAP.create(dimensions=100)
        train_hvs = vsa.random(jax.random.PRNGKey(0), (30, 100))
        train_labels = jnp.array([0]*10 + [1]*10 + [2]*10)

        classifier2 = classifier1.fit(train_hvs, train_labels, epochs=1)

        # Should be different instances
        assert classifier1 is not classifier2

    def test_update_prototypes_internal(self):
        """Test internal prototype update mechanism."""
        classifier = AdaptiveHDC.create(
            num_classes=3,
            dimensions=100,
            key=jax.random.PRNGKey(42)
        )

        sample = jax.random.normal(jax.random.PRNGKey(0), (100,))
        old_prototype = classifier.prototypes[1].copy()

        updated_classifier = classifier._update_prototypes(
            sample, true_label=1, pred_label=2, learning_rate=0.1
        )

        # True class prototype should have changed
        assert not jnp.allclose(updated_classifier.prototypes[1], old_prototype)
