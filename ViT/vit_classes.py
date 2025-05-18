import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

# Include all your custom layer definitions
class PatchCreation(layers.Layer):
    def __init__(self, patch_size, trainable=False, **kwargs):
        super(PatchCreation, self).__init__(trainable=trainable, **kwargs)
        self.patch_size = patch_size
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def get_config(self):
        config = super(PatchCreation, self).get_config()
        config.update({
            'patch_size': self.patch_size
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class PatchEmbedding(layers.Layer):
    def __init__(self, num_patches, projection_dim, trainable=False, **kwargs):
        super(PatchEmbedding, self).__init__(trainable=trainable, **kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(projection_dim)
        
    def call(self, patches):
        return self.projection(patches)

    def get_config(self):
        config = super(PatchEmbedding, self).get_config()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class PositionEmbedding(layers.Layer):
    def __init__(self, num_patches, projection_dim, trainable=False, **kwargs):
        super(PositionEmbedding, self).__init__(trainable=trainable, **kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        
    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super(PositionEmbedding, self).get_config()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, trainable=True, **kwargs):
        super(TransformerEncoderBlock, self).__init__(trainable=trainable, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        # Removed dropout from MultiHeadAttention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Removed dropout from MLP
        self.mlp = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim)
        ])
    
    def call(self, inputs, training=True):
        # Apply Layer Normalization before attention
        x1 = self.layernorm1(inputs)
        attention_output = self.attention(x1, x1, training=training)
        x2 = attention_output + inputs
        
        # Apply Layer Normalization before MLP
        x3 = self.layernorm2(x2)
        x4 = self.mlp(x3, training=training)
        return x2 + x4
        
    def get_config(self):
        config = super(TransformerEncoderBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)