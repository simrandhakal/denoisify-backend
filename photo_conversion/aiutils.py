from tensorflow.keras.layers import Layer, Conv2D, LeakyReLU, Add
from tensorflow.keras import backend as K
import tensorflow as tf

def residual_block(x, filters):
    res = Conv2D(filters, kernel_size=3, strides=1, padding="same")(x)
    res = InstanceNormalization()(res)
    res = LeakyReLU(alpha=0.2)(res)

    res = Conv2D(filters, kernel_size=3, strides=1, padding="same")(res)
    res = InstanceNormalization()(res)

    return Add()([x, res])  # Skip Connection

# Custom Layer for Instance Normalization
@tf.keras.utils.register_keras_serializable()
class InstanceNormalization(tf.keras.layers.Layer):
    """
    Implements Instance Normalization as a custom Keras layer.
    
    Instance Normalization normalizes each individual sample (instance) 
    across its spatial dimensions (height & width), making it useful 
    in tasks like image generation where batch statistics can be unstable.
    """

    def __init__(self, epsilon=1e-5, name=None, **kwargs):
        """
        Initializes the InstanceNormalization layer.

        Parameters:
        - epsilon (float): A small constant added to the variance to avoid division by zero.
        """
        super(InstanceNormalization, self).__init__(name=name, **kwargs)
        self.epsilon = epsilon  # Small value to stabilize normalization

    def call(self, x):
        """
        Performs instance normalization on the input tensor.

        Parameters:
        - x (Tensor): Input feature map of shape (batch, height, width, channels).

        Returns:
        - Tensor: Normalized output of the same shape as input.
        """
        # Compute the mean and variance along spatial dimensions (height & width)
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)

        # Normalize the input using computed mean & variance
        return (x - mean) / tf.sqrt(var + self.epsilon)
    
    def get_config(self):
        """
        Returns the configuration of the layer for serialization.
        """
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

    @classmethod
    def from_config(cls, config):
        """
        Recreates the layer from the saved configuration.
        """
        return cls(**config)
    

@tf.keras.utils.register_keras_serializable()
class SpectralNormalization(Layer):
    """
    Spectral Normalization layer for stabilizing GAN training.
    
    - Normalizes the weight matrix using power iteration.
    - Supports Conv2D, Dense, and Conv2DTranspose layers.
    """

    def __init__(self, layer, power_iterations=1, **kwargs):
        """
        Initializes SpectralNormalization layer.

        Parameters:
        - layer: The base layer (Conv2D, Dense, etc.) to apply normalization to.
        - power_iterations: Number of power iterations to approximate the spectral norm.
        """
        super(SpectralNormalization, self).__init__(**kwargs)
        self.layer = layer
        self.power_iterations = power_iterations
        

    def build(self, input_shape):
        """
        Builds the layer by initializing spectral normalization weights.
        """
        if not hasattr(self.layer, 'kernel'):
            raise ValueError("SpectralNormalization must be applied to layers with a `kernel` attribute.")

        # Build the wrapped layer
        self.layer.build(input_shape)
        self.kernel = self.layer.kernel  # Get the original kernel (weights)

        # Create 'u' for spectral normalization
        self.u = self.add_weight(
            shape=(1, self.kernel.shape[-1]),  # Shape: (1, output_channels)
            initializer="random_normal",
            trainable=False,
            name="sn_u"
        )
        super(SpectralNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        Applies spectral normalization during training.
        """
        w_reshaped = tf.reshape(self.kernel, [-1, self.kernel.shape[-1]])  # Reshape to 2D matrix
        u_hat = self.u

        # Power iteration for spectral norm estimation
        for _ in range(self.power_iterations):
            v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(w_reshaped)))
            u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w_reshaped))

        # Detach u_hat and v_hat from the computational graph
        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        # Compute spectral norm
        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))

        # Normalize weights
        w_sn = w_reshaped / sigma
        w_sn = tf.reshape(w_sn, self.kernel.shape)

        # Assign updated u for next forward pass
        self.u.assign(u_hat)

        # Replace kernel with the normalized version
        self.layer.kernel.assign(w_sn)

        return self.layer(inputs)

    def compute_output_shape(self, input_shape):
        """
        Returns the output shape of the layer.
        """
        return self.layer.compute_output_shape(input_shape)
    
    def get_config(self):
        """
        Returns the configuration of the layer for serialization.
        """
        config = super().get_config()
        config.update({
            "power_iterations": self.power_iterations,
            "layer": self.layer
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Recreates the layer from the saved configuration.
        """
        return cls(**config)

