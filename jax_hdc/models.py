"""Classification and learning models for Hyperdimensional Computing.

This module provides various HDC learning algorithms including centroid-based
classification, learning vector quantization, and gradient-based methods.
"""

from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Optional, Union, Callable, Any, List
import jax
import jax.numpy as jnp
from jax_hdc import functional as F
from jax_hdc.vsa import VSAModel, create_vsa_model


@jax.tree_util.register_dataclass
@dataclass
class CentroidClassifier:
    """Centroid-based classifier for Hyperdimensional Computing.

    Stores one prototype (centroid) hypervector per class. Classification
    is performed by finding the most similar prototype to the query.

    Properties:
        - Simple and interpretable
        - Fast training (single pass)
        - Online learning capable
        - No hyperparameters

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from jax_hdc import MAP, CentroidClassifier
        >>>
        >>> # Create classifier
        >>> model = MAP.create(dimensions=10000)
        >>> classifier = CentroidClassifier.create(
        ...     num_classes=10,
        ...     dimensions=10000,
        ...     vsa_model=model
        ... )
        >>>
        >>> # Train on encoded data
        >>> key = jax.random.PRNGKey(42)
        >>> train_hvs = model.random(key, (100, 10000))  # 100 samples
        >>> train_labels = jax.random.randint(key, (100,), 0, 10)  # 10 classes
        >>> classifier = classifier.fit(train_hvs, train_labels)
        >>>
        >>> # Predict
        >>> test_hvs = model.random(key, (20, 10000))
        >>> predictions = classifier.predict(test_hvs)
        >>> print(predictions.shape)
        (20,)
    """

    # Data fields
    prototypes: jax.Array  # Shape: (num_classes, dimensions)

    # Metadata fields
    num_classes: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))
    vsa_model_name: str = field(metadata=dict(static=True), default="map")

    @staticmethod
    def create(
        num_classes: int,
        dimensions: int = 10000,
        vsa_model: Union[str, VSAModel] = "map",
        initial_prototypes: Optional[jax.Array] = None,
        key: Optional[jax.Array] = None
    ) -> "CentroidClassifier":
        """Create a centroid classifier.

        Args:
            num_classes: Number of classes
            dimensions: Dimensionality of hypervectors (default: 10000)
            vsa_model: VSA model to use ('bsc', 'map', 'hrr', 'fhrr') or VSAModel instance
            initial_prototypes: Optional initial prototypes of shape (num_classes, dimensions)
            key: JAX random key for initialization

        Returns:
            Initialized CentroidClassifier
        """
        # Handle both string and VSAModel
        if isinstance(vsa_model, str):
            vsa_model_name = vsa_model
            vsa = create_vsa_model(vsa_model, dimensions)
        else:
            vsa_model_name = vsa_model.name
            vsa = vsa_model

        # Initialize prototypes
        if initial_prototypes is not None:
            prototypes = initial_prototypes
        else:
            if key is None:
                key = jax.random.PRNGKey(0)
            # Initialize with random hypervectors
            prototypes = vsa.random(key, shape=(num_classes, dimensions))

        return CentroidClassifier(
            prototypes=prototypes,
            num_classes=num_classes,
            dimensions=dimensions,
            vsa_model_name=vsa_model_name
        )

    @jax.jit
    def similarity(self, query: jax.Array) -> jax.Array:
        """Compute similarity between query and all class prototypes.

        Args:
            query: Query hypervector of shape (dimensions,)

        Returns:
            Similarity scores of shape (num_classes,)
        """
        if self.vsa_model_name == "bsc":
            return jax.vmap(lambda p: F.hamming_similarity(query, p))(self.prototypes)
        else:
            return jax.vmap(lambda p: F.cosine_similarity(query, p))(self.prototypes)

    @jax.jit
    def predict(self, queries: jax.Array) -> jax.Array:
        """Predict class labels for queries.

        Args:
            queries: Query hypervectors of shape (batch_size, dimensions)
                    or single query of shape (dimensions,)

        Returns:
            Predicted class indices of shape (batch_size,) or scalar
        """
        # Handle both single and batch queries
        is_single = queries.ndim == 1
        if is_single:
            queries = queries[None, :]

        # Compute similarities and find best match
        similarities = jax.vmap(self.similarity)(queries)
        predictions = jnp.argmax(similarities, axis=-1)

        if is_single:
            return predictions[0]
        return predictions

    @jax.jit
    def predict_proba(self, queries: jax.Array) -> jax.Array:
        """Predict class probabilities using softmax of similarities.

        Args:
            queries: Query hypervectors of shape (batch_size, dimensions)
                    or single query of shape (dimensions,)

        Returns:
            Class probabilities of shape (batch_size, num_classes) or (num_classes,)
        """
        # Handle both single and batch queries
        is_single = queries.ndim == 1
        if is_single:
            queries = queries[None, :]

        # Compute similarities
        similarities = jax.vmap(self.similarity)(queries)

        # Convert to probabilities via softmax
        probs = jax.nn.softmax(similarities, axis=-1)

        if is_single:
            return probs[0]
        return probs

    def fit(self, train_hvs: jax.Array, train_labels: jax.Array) -> "CentroidClassifier":
        """Train classifier by computing class centroids.

        Args:
            train_hvs: Training hypervectors of shape (n_samples, dimensions)
            train_labels: Training labels of shape (n_samples,) with values in [0, num_classes)

        Returns:
            Trained CentroidClassifier (new instance, immutable)
        """
        # Compute centroid for each class using where instead of boolean indexing
        new_prototypes_list: List[jax.Array] = []

        for class_idx in range(self.num_classes):
            # Get all hypervectors for this class
            class_mask = train_labels == class_idx

            # Count samples in this class
            num_samples = jnp.sum(class_mask)

            if num_samples > 0:
                # Use where to select class samples (more JIT-friendly)
                # Create weights: 1.0 for class samples, 0.0 for others
                weights = jnp.where(class_mask[:, None], 1.0, 0.0)

                # Weighted sum of hypervectors
                if self.vsa_model_name == "bsc":
                    # For BSC, use weighted voting
                    weighted_hvs = train_hvs.astype(jnp.float32) * weights
                    summed = jnp.sum(weighted_hvs, axis=0)
                    threshold = num_samples / 2.0
                    centroid = summed > threshold
                else:
                    # For real-valued, use weighted average
                    weighted_hvs = train_hvs * weights
                    summed = jnp.sum(weighted_hvs, axis=0)
                    # Normalize
                    norm = jnp.linalg.norm(summed)
                    centroid = summed / (norm + 1e-8)

                new_prototypes_list.append(centroid)
            else:
                # Keep initial random prototype for empty classes
                new_prototypes_list.append(self.prototypes[class_idx])

        new_prototypes: jax.Array = jnp.stack(new_prototypes_list)

        # Return new instance with updated prototypes
        return self.replace(prototypes=new_prototypes)

    def update_online(
        self,
        sample_hv: jax.Array,
        label: int,
        learning_rate: float = 0.1
    ) -> "CentroidClassifier":
        """Update classifier online with a single sample.

        Args:
            sample_hv: Sample hypervector of shape (dimensions,)
            label: True label (integer in [0, num_classes))
            learning_rate: Learning rate for prototype update (default: 0.1)

        Returns:
            Updated CentroidClassifier
        """
        # Update the prototype for this class
        old_prototype = self.prototypes[label]

        if self.vsa_model_name == "bsc":
            # For binary, use weighted bundling
            combined = jnp.stack([old_prototype, sample_hv])
            new_prototype = F.bundle_bsc(combined, axis=0)
        else:
            # For real-valued, use moving average
            new_prototype = (1 - learning_rate) * old_prototype + learning_rate * sample_hv
            # Normalize
            norm = jnp.linalg.norm(new_prototype)
            new_prototype = new_prototype / (norm + 1e-8)

        # Update prototypes array
        new_prototypes = self.prototypes.at[label].set(new_prototype)

        return self.replace(prototypes=new_prototypes)

    @jax.jit
    def score(self, test_hvs: jax.Array, test_labels: jax.Array) -> jax.Array:
        """Compute accuracy on test data.

        Args:
            test_hvs: Test hypervectors of shape (n_samples, dimensions)
            test_labels: True labels of shape (n_samples,)

        Returns:
            Accuracy score in [0, 1] as a JAX scalar array
        """
        predictions = self.predict(test_hvs)
        return jnp.mean(predictions == test_labels)

    def replace(self, **updates: Any) -> "CentroidClassifier":
        """Create a new instance with updated fields (dataclass pattern)."""
        return dataclass_replace(self, **updates)


@jax.tree_util.register_dataclass
@dataclass
class AdaptiveHDC:
    """Adaptive HDC classifier with prototype refinement.

    Extends centroid classifier with iterative refinement of prototypes
    based on misclassified samples.

    Properties:
        - Improves on centroid baseline
        - Iterative training
        - Handles difficult samples better

    Example:
        >>> import jax
        >>> from jax_hdc import MAP, AdaptiveHDC
        >>>
        >>> model = MAP.create(dimensions=10000)
        >>> classifier = AdaptiveHDC.create(
        ...     num_classes=10,
        ...     dimensions=10000,
        ...     vsa_model=model
        ... )
        >>> # Train with multiple epochs
        >>> key = jax.random.PRNGKey(42)
        >>> train_hvs = model.random(key, (100, 10000))
        >>> train_labels = jax.random.randint(key, (100,), 0, 10)
        >>> classifier = classifier.fit(train_hvs, train_labels, epochs=10)
    """

    # Data fields
    prototypes: jax.Array
    num_updates: jax.Array  # Track number of updates per class

    # Metadata fields
    num_classes: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))
    vsa_model_name: str = field(metadata=dict(static=True), default="map")

    @staticmethod
    def create(
        num_classes: int,
        dimensions: int = 10000,
        vsa_model: Union[str, VSAModel] = "map",
        key: Optional[jax.Array] = None
    ) -> "AdaptiveHDC":
        """Create an adaptive HDC classifier."""
        if isinstance(vsa_model, str):
            vsa_model_name = vsa_model
            vsa = create_vsa_model(vsa_model, dimensions)
        else:
            vsa_model_name = vsa_model.name
            vsa = vsa_model

        if key is None:
            key = jax.random.PRNGKey(0)

        prototypes = vsa.random(key, shape=(num_classes, dimensions))
        num_updates = jnp.zeros(num_classes, dtype=jnp.int32)

        return AdaptiveHDC(
            prototypes=prototypes,
            num_updates=num_updates,
            num_classes=num_classes,
            dimensions=dimensions,
            vsa_model_name=vsa_model_name
        )

    @jax.jit
    def predict(self, queries: jax.Array) -> jax.Array:
        """Predict class labels."""
        is_single = queries.ndim == 1
        if is_single:
            queries = queries[None, :]

        if self.vsa_model_name == "bsc":
            similarities = jax.vmap(
                lambda q: jax.vmap(lambda p: F.hamming_similarity(q, p))(self.prototypes)
            )(queries)
        else:
            similarities = jax.vmap(
                lambda q: jax.vmap(lambda p: F.cosine_similarity(q, p))(self.prototypes)
            )(queries)

        predictions = jnp.argmax(similarities, axis=-1)

        if is_single:
            return predictions[0]
        return predictions

    def fit(
        self,
        train_hvs: jax.Array,
        train_labels: jax.Array,
        epochs: int = 1,
        learning_rate: float = 0.1
    ) -> "AdaptiveHDC":
        """Train with iterative refinement.

        Args:
            train_hvs: Training hypervectors
            train_labels: Training labels
            epochs: Number of training epochs (default: 1)
            learning_rate: Learning rate for updates (default: 0.1)

        Returns:
            Trained classifier
        """
        # Initialize with centroids using where instead of boolean indexing
        classifier = self
        for class_idx in range(self.num_classes):
            class_mask = train_labels == class_idx
            num_samples = jnp.sum(class_mask)

            if num_samples > 0:
                # Use where to select class samples (more JIT-friendly)
                weights = jnp.where(class_mask[:, None], 1.0, 0.0)

                if self.vsa_model_name == "bsc":
                    # For BSC, use weighted voting
                    weighted_hvs = train_hvs.astype(jnp.float32) * weights
                    summed = jnp.sum(weighted_hvs, axis=0)
                    threshold = num_samples / 2.0
                    centroid = summed > threshold
                else:
                    # For real-valued, use weighted average
                    weighted_hvs = train_hvs * weights
                    summed = jnp.sum(weighted_hvs, axis=0)
                    norm = jnp.linalg.norm(summed)
                    centroid = summed / (norm + 1e-8)

                classifier = classifier.replace(
                    prototypes=classifier.prototypes.at[class_idx].set(centroid)
                )

        # Iterative refinement
        for epoch in range(epochs):
            for i in range(len(train_hvs)):
                pred = classifier.predict(train_hvs[i])
                true_label = train_labels[i]

                # Update if misclassified
                if pred != true_label:
                    classifier = classifier._update_prototypes(
                        train_hvs[i], true_label, pred, learning_rate
                    )

        return classifier

    def _update_prototypes(
        self,
        sample_hv: jax.Array,
        true_label: Union[int, jax.Array],
        pred_label: Union[int, jax.Array],
        learning_rate: float
    ) -> "AdaptiveHDC":
        """Update prototypes based on misclassification."""
        # Move true class prototype towards sample
        true_proto = self.prototypes[true_label]

        if self.vsa_model_name != "bsc":
            new_true_proto = true_proto + learning_rate * sample_hv
            new_true_proto = new_true_proto / (jnp.linalg.norm(new_true_proto) + 1e-8)
        else:
            new_true_proto = F.bundle_bsc(jnp.stack([true_proto, sample_hv]), axis=0)

        new_prototypes = self.prototypes.at[true_label].set(new_true_proto)

        return self.replace(prototypes=new_prototypes)

    @jax.jit
    def score(self, test_hvs: jax.Array, test_labels: jax.Array) -> jax.Array:
        """Compute accuracy as JAX scalar array."""
        predictions = self.predict(test_hvs)
        return jnp.mean(predictions == test_labels)

    def replace(self, **updates: Any) -> "AdaptiveHDC":
        """Create new instance with updates."""
        return dataclass_replace(self, **updates)


__all__ = [
    "CentroidClassifier",
    "AdaptiveHDC",
]
