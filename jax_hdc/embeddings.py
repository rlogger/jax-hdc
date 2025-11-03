"""Encoders for transforming data into hypervectors.

This module provides various encoding strategies to transform different types
of data (discrete features, continuous values, images) into hypervectors.
"""

from dataclasses import dataclass, field
from typing import Optional, Union
import jax
import jax.numpy as jnp
from jax_hdc import functional as F
from jax_hdc.vsa import VSAModel, create_vsa_model

@jax.tree_util.register_dataclass
@dataclass
class RandomEncoder:
    """Encoder using random hypervectors for discrete features.

    Each unique feature value is mapped to a random hypervector from a codebook.
    Multiple features are bundled together to form the final representation.

    Properties:
        - Suitable for categorical/discrete data
        - Preserves similarity: similar feature values map to similar hypervectors
        - Memory efficient: stores only the codebook

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> # Create encoder for 10 features with 100 possible values each
        >>> encoder = RandomEncoder.create(
        ...     num_features=10,
        ...     num_values=100,
        ...     dimensions=10000,
        ...     vsa_model='map',
        ...     key=jax.random.PRNGKey(42)
        ... )
        >>> # Encode a sample with feature indices
        >>> sample = jnp.array([5, 12, 3, 89, 45, 67, 23, 8, 91, 34])
        >>> encoded = encoder.encode(sample)
        >>> print(encoded.shape)
        (10000,)
    """

    # Data fields (traced by JAX)
    codebook: jax.Array  # Shape: (num_features, num_values, dimensions)

    # Metadata fields (static, not traced)
    num_features: int = field(metadata=dict(static=True))
    num_values: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))
    vsa_model_name: str = field(metadata=dict(static=True), default="map")

    @staticmethod
    def create(
        num_features: int,
        num_values: int,
        dimensions: int = 10000,
        vsa_model: Union[str, VSAModel] = "map",
        key: Optional[jax.Array] = None
    ) -> "RandomEncoder":
        """Create a random encoder.

        Args:
            num_features: Number of features to encode
            num_values: Number of possible values per feature
            dimensions: Dimensionality of hypervectors (default: 10000)
            vsa_model: VSA model to use ('bsc', 'map', 'hrr', 'fhrr') or VSAModel instance
            key: JAX random key (default: PRNGKey(0))

        Returns:
            Initialized RandomEncoder
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        # Handle both string and VSAModel
        if isinstance(vsa_model, str):
            vsa_model_name = vsa_model
            vsa = create_vsa_model(vsa_model, dimensions)
        else:
            vsa_model_name = vsa_model.name
            vsa = vsa_model

        # Generate random codebook
        codebook = vsa.random(key, shape=(num_features, num_values, dimensions))

        return RandomEncoder(
            codebook=codebook,
            num_features=num_features,
            num_values=num_values,
            dimensions=dimensions,
            vsa_model_name=vsa_model_name
        )

    @jax.jit
    def encode(self, indices: jax.Array) -> jax.Array:
        """Encode discrete features as hypervectors.

        Args:
            indices: Feature indices of shape (num_features,) with values in [0, num_values)

        Returns:
            Encoded hypervector of shape (dimensions,)
        """
        # Select hypervector for each feature
        # codebook[i, indices[i]] selects the hypervector for feature i with value indices[i]
        selected = jax.vmap(lambda i: self.codebook[i, indices[i]])(
            jnp.arange(self.num_features)
        )

        # Bundle all feature hypervectors
        if self.vsa_model_name == "bsc":
            return F.bundle_bsc(selected, axis=0)
        else:
            return F.bundle_map(selected, axis=0)

    @jax.jit
    def encode_batch(self, indices: jax.Array) -> jax.Array:
        """Encode a batch of samples.

        Args:
            indices: Batch of feature indices of shape (batch_size, num_features)

        Returns:
            Encoded hypervectors of shape (batch_size, dimensions)
        """
        return jax.vmap(self.encode)(indices)


@jax.tree_util.register_dataclass
@dataclass
class LevelEncoder:
    """Encoder for continuous values using level hypervectors.

    Continuous values are encoded by interpolating between level hypervectors,
    creating a smooth representation where similar values map to similar hypervectors.

    Two encoding strategies:
    - Thermometer: Binary encoding with cumulative levels
    - Circular: Smooth interpolation for periodic values

    Properties:
        - Preserves ordering: similar values → similar hypervectors
        - Differentiable: compatible with gradient-based learning
        - Configurable resolution

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> # Create encoder for values in range [0, 1]
        >>> encoder = LevelEncoder.create(
        ...     num_levels=100,
        ...     dimensions=10000,
        ...     min_value=0.0,
        ...     max_value=1.0,
        ...     vsa_model='map',
        ...     key=jax.random.PRNGKey(42)
        ... )
        >>> # Encode a continuous value
        >>> value = 0.75
        >>> encoded = encoder.encode(value)
        >>> print(encoded.shape)
        (10000,)
    """

    # Data fields
    level_hvs: jax.Array  # Shape: (num_levels, dimensions)

    # Metadata fields
    num_levels: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))
    min_value: float = field(metadata=dict(static=True))
    max_value: float = field(metadata=dict(static=True))
    vsa_model_name: str = field(metadata=dict(static=True), default="map")
    encoding_type: str = field(metadata=dict(static=True), default="linear")

    @staticmethod
    def create(
        num_levels: int = 100,
        dimensions: int = 10000,
        min_value: float = 0.0,
        max_value: float = 1.0,
        vsa_model: Union[str, VSAModel] = "map",
        encoding_type: str = "linear",
        key: Optional[jax.Array] = None
    ) -> "LevelEncoder":
        """Create a level encoder.

        Args:
            num_levels: Number of levels for discretization (default: 100)
            dimensions: Dimensionality of hypervectors (default: 10000)
            min_value: Minimum value of the range (default: 0.0)
            max_value: Maximum value of the range (default: 1.0)
            vsa_model: VSA model to use ('bsc', 'map', 'hrr', 'fhrr')
            encoding_type: 'linear' or 'circular' (default: 'linear')
            key: JAX random key

        Returns:
            Initialized LevelEncoder
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        # Handle both string and VSAModel
        if isinstance(vsa_model, str):
            vsa_model_name = vsa_model
            vsa = create_vsa_model(vsa_model, dimensions)
        else:
            vsa_model_name = vsa_model.name
            vsa = vsa_model

        # Generate random level hypervectors
        level_hvs = vsa.random(key, shape=(num_levels, dimensions))

        return LevelEncoder(
            level_hvs=level_hvs,
            num_levels=num_levels,
            dimensions=dimensions,
            min_value=min_value,
            max_value=max_value,
            vsa_model_name=vsa_model_name,
            encoding_type=encoding_type
        )

    @jax.jit
    def encode(self, value: Union[float, jax.Array]) -> jax.Array:
        """Encode a continuous value as a hypervector.

        Args:
            value: Continuous value to encode (scalar or array)

        Returns:
            Encoded hypervector of shape (dimensions,) or batch shape + (dimensions,)
        """
        # Normalize value to [0, num_levels - 1]
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        normalized = jnp.clip(normalized, 0.0, 1.0)
        level_pos = normalized * (self.num_levels - 1)

        # Get lower and upper level indices
        lower_idx = jnp.floor(level_pos).astype(jnp.int32)
        upper_idx = jnp.ceil(level_pos).astype(jnp.int32)

        # Interpolation weight
        weight = level_pos - lower_idx

        # Get level hypervectors
        lower_hv = self.level_hvs[lower_idx]
        upper_hv = self.level_hvs[upper_idx]

        # Linear interpolation for real-valued models
        if self.vsa_model_name in ["map", "hrr", "fhrr"]:
            # Weighted combination
            encoded = (1 - weight[..., None]) * lower_hv + weight[..., None] * upper_hv
            # Normalize
            norm = jnp.linalg.norm(encoded, axis=-1, keepdims=True)
            return encoded / (norm + 1e-8)
        else:  # BSC
            # For binary, use threshold-based selection
            return jnp.where(weight[..., None] > 0.5, upper_hv, lower_hv)

    @jax.jit
    def encode_batch(self, values: jax.Array) -> jax.Array:
        """Encode a batch of continuous values.

        Args:
            values: Batch of values of shape (batch_size,) or (batch_size, num_features)

        Returns:
            Encoded hypervectors
        """
        return jax.vmap(self.encode)(values)


@jax.tree_util.register_dataclass
@dataclass
class ProjectionEncoder:
    """Encoder using random projection for high-dimensional data.

    Projects high-dimensional input data into hypervector space using a
    random projection matrix. Useful for images, text embeddings, etc.

    Properties:
        - Preserves approximate distances (Johnson-Lindenstrauss lemma)
        - Efficient for high-dimensional data
        - No training required

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> # Create encoder for 784-dimensional input (e.g., 28x28 image)
        >>> encoder = ProjectionEncoder.create(
        ...     input_dim=784,
        ...     dimensions=10000,
        ...     vsa_model='map',
        ...     key=jax.random.PRNGKey(42)
        ... )
        >>> # Encode a flattened image
        >>> image = jax.random.normal(jax.random.PRNGKey(0), (784,))
        >>> encoded = encoder.encode(image)
        >>> print(encoded.shape)
        (10000,)
    """

    # Data fields
    projection_matrix: jax.Array  # Shape: (input_dim, dimensions)

    # Metadata fields
    input_dim: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))
    vsa_model_name: str = field(metadata=dict(static=True), default="map")

    @staticmethod
    def create(
        input_dim: int,
        dimensions: int = 10000,
        vsa_model: Union[str, VSAModel] = "map",
        key: Optional[jax.Array] = None
    ) -> "ProjectionEncoder":
        """Create a projection encoder.

        Args:
            input_dim: Dimensionality of input data
            dimensions: Dimensionality of hypervectors (default: 10000)
            vsa_model: VSA model to use ('bsc', 'map', 'hrr', 'fhrr')
            key: JAX random key

        Returns:
            Initialized ProjectionEncoder
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        # Handle both string and VSAModel
        if isinstance(vsa_model, str):
            vsa_model_name = vsa_model
        else:
            vsa_model_name = vsa_model.name

        # Create random projection matrix (normalized)
        projection_matrix = jax.random.normal(key, shape=(input_dim, dimensions))
        projection_matrix = projection_matrix / jnp.sqrt(input_dim)

        return ProjectionEncoder(
            projection_matrix=projection_matrix,
            input_dim=input_dim,
            dimensions=dimensions,
            vsa_model_name=vsa_model_name
        )

    @jax.jit
    def encode(self, x: jax.Array) -> jax.Array:
        """Encode input data as a hypervector.

        Args:
            x: Input data of shape (input_dim,)

        Returns:
            Encoded hypervector of shape (dimensions,)
        """
        # Random projection
        projected = jnp.dot(x, self.projection_matrix)

        # Apply activation based on VSA model
        if self.vsa_model_name == "bsc":
            # Threshold for binary
            return projected > 0
        else:
            # Normalize for real-valued
            norm = jnp.linalg.norm(projected)
            return projected / (norm + 1e-8)

    @jax.jit
    def encode_batch(self, x: jax.Array) -> jax.Array:
        """Encode a batch of inputs.

        Args:
            x: Batch of inputs of shape (batch_size, input_dim)

        Returns:
            Encoded hypervectors of shape (batch_size, dimensions)
        """
        return jax.vmap(self.encode)(x)


__all__ = [
    "RandomEncoder",
    "LevelEncoder",
    "ProjectionEncoder",
]
