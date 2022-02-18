import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import *
from keras.utils import conv_utils
import numpy as np

########################################################################################################################
EPSILON = tf.keras.backend.epsilon()


########################################################################################################################
def Multi_Head_Attention(num_heads, embedding_dims):
    def separate_heads(x): # B, n_patch, embedding_dims
        assert x.shape[1] % num_heads == 0
        x = tf.reshape(x, [-1, x.shape[1]//num_heads, num_heads, x.shape[-1]])
        return tf.transpose(x, perm=[0, 2, 1, 3])  # B, n_heads, n_patch/n_heads, embedding_dims

    def self_attention(Q, K, V): # B, n_heads, n_patch/n_heads, embedding_dims
        score = tf.matmul(Q, K, transpose_b=True)  # B, n_head, n_patch/n_heads, n_patch/n_heads
        dim_key = tf.cast(K.shape[-1], tf.float32) # emb_dims
        scaled_score = score / tf.math.sqrt(dim_key) + EPSILON
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, V) # B, n_heads, n_patch/n_heads, embedding_dims
        return output, weights

    def main(x):
        Q, K, V = [Dense(embedding_dims)(x) for _ in range(3)]
        Q, K, V = [separate_heads(target) for target in [Q, K, V]]

        x, weights = self_attention(Q, K, V)
        x = tf.transpose(x, [0, 2, 1, 3])  # B, n_patch/n_heads, n_heads, embedding_dims
        x = tf.reshape(x, [-1, x.shape[1]*x.shape[2], embedding_dims]) # B, n_patch, embedding_dims
        return Dense(embedding_dims)(x)
    return main

def transformer_encoder(num_heads, embed_dims, drop_rate=0.2):
    def main(inputs):
        x = LayerNormalization()(inputs)
        x = Multi_Head_Attention(num_heads, embed_dims)(x) # B, n_patch, emb_dims
        x = Dropout(0.2)(x)
        x = x + inputs # B, n_patch, emb_dims

        x_ = LayerNormalization()(x)
        x_ = MLP([embed_dims*2, embed_dims])(x_)
        x_ = Dropout(drop_rate)(x_)
        return x + x_ # B, n_patch, embed_dims
    return main

def MLP(units, drop_rate=0.2):
    units = list(units)
    def main(x):
        for unit in units:
            x = Dense(unit, 'gelu')(x)
            x = Dropout(drop_rate)(x)
        return x
    return main

########################################################################################################################
def patch_encoding(embed_dims):
    def main(x):
        x = Dense(units=embed_dims)(x)
        position_embed = Embedding(input_dim=x.shape[1], output_dim=embed_dims)
        return x + position_embed(tf.range(x.shape[1]))
    return main

def make_patches(patch_size):
    def main(x):
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        shape = [-1, patches.shape[1]*patches.shape[2], patches.shape[-1]]
        return tf.reshape(patches, shape)
    return main
########################################################################################################################
def ViT(num_classes, patch_size, num_encodings=8, num_heads=4, embed_dims=64):
    encodings = [transformer_encoder(num_heads, embed_dims) for _ in range(num_encodings)]
    def main(x):
        ### Patch
        x = make_patches(patch_size)(x)
        x = patch_encoding(embed_dims)(x) # B, n_patch, embed_dims
        ### Transformer Encoding
        for enc in encodings:
            x = enc(x) # B, n_patch, embed_dims
        ###
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = MLP([2048, 1024], drop_rate=0.5)(x)
        output = Dense(num_classes)(x)
        return output
    return main

########################################################################################################################
