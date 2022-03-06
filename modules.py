from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np

########################################################################################################################
EPSILON = tf.keras.backend.epsilon()
TruncatedNormal = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
NA = tf.newaxis

########################################################################################################################
def patch_partition(emb_dim, patch_size, drop_rate=0.2):
    conv_stem = Conv2D(emb_dim, kernel_size=patch_size, strides=patch_size)
    def main(x):
        # _, H, W, C = x.shape
        x = conv_stem(x)
        # x = tf.reshape(x, [-1, H*W, C])
        x = LayerNormalization(epsilon=EPSILON)(x)
        x = Dropout(drop_rate)(x)
        return x
    return main

def patch_merging(size=2):
    def main(x):
        _, H, W, C = x.shape
        sizes = [1, size, size, 1]
        x = tf.image.extract_patches(x, sizes, sizes, [1,1,1,1], 'VALID')
        """
        ### patches rearrange
        shape = [-1] + x.shape.as_list()[1:] # B, nP, nP, sz*sz*C
        x = tf.split(x, x.shape[-1] // C, -1)
        x = tf.stack(x, -1)
        x = tf.reshape(x, shape)
        """
        # _, H_, W_, C_ = x.shape.as_list()
        # x = tf.reshape(x, [-1, H_*W_, C_])
        x = LayerNormalization(epsilon=EPSILON)(x)
        x = Dense(C*2, kernel_initializer=TruncatedNormal, use_bias=False)(x)
        return x
    return main

def window_partitioning(wsz):
    assert wsz > 1
    def window_reverse(wsz):
        def main(x):
            _, H, W, C = x.shape
            x = tf.reshape(x, [-1, H // wsz, W // wsz, wsz, wsz, C])
            x = tf.transpose(x, [0, 1, 3, 2, 4, 5]) # B, H // wsz, wsz, W // wsz, wsz, C
            x = tf.reshape(x, [-1, H, W, C])
            return x
        return main

    def window_partition(wsz):
        def main(x):
            _, H, W, C = x.shape
            x = tf.reshape(x, [-1, H // wsz, wsz, W // wsz, wsz, C])
            x = tf.transpose(x, [0, 1, 3, 2, 4, 5])  # B, H/w, W/w, w, w, C
            x = tf.reshape(x, [-1, wsz, wsz, C])  # (B*H/w*W/w), w, w, C
            return x
        return main

    def wrapper(function):
        def main(x, mask=None):
            if wsz <= 1:
                return function(x, mask)
            _, H, W, C = x.shape
            x = window_partition(wsz)(x)
            x = function(x, mask) # B, I, C
            x = window_reverse(wsz)(x)
            return x
        return main
    return wrapper

def shift(shift_size):
    def wrapper(function):
        def main(x, mask=None):
            if shift_size <= 0:
                return function(x, mask)
            x = tf.roll(x, [-shift_size, -shift_size], [1, 2])
            x = function(x, mask)
            x = tf.roll(x, [shift_size, shift_size], [1, 2])
            return x
        return main
    return wrapper

def wMSA(emb_dim, num_heads, window_size=7, shift_size=3, drop_rate=0.2):
    """
    relative position 값을 더하고 masking을 한다는 점이 다름

    * input shape = [B, H, W, C]
    :return shape = [B, H, W, C]
    """
    def get_relative_position(H, W):
        x, y = [i for i in range(0, H)], [i for i in range(0, W)]
        x = tf.cast(tf.meshgrid(x, y), tf.float32)
        x = tf.reshape(tf.stack(x), [2, -1])
        x = x[:, :, None] - x[:, None, :]
        x = tf.transpose(x, [1, 2, 0])

        a = tf.constant(W - 1, dtype=x.dtype, shape=x.shape[:-1])
        b = tf.constant(H - 1, dtype=x.dtype, shape=x.shape[:-1])
        x = x + tf.stack([a, b], -1)

        a = tf.ones(a.shape)
        b = tf.constant(2 * H - 1, dtype=x.dtype, shape=a.shape)
        x = x * tf.stack([a, b], -1)
        x = tf.reduce_sum(x, -1)
        return x

    def set_mask(x, mask):
        nW, I, I = mask.shape # nWindow, wsz*wsz, wsz*wsz
        x = tf.reshape(x, [-1, nW, num_heads, I, I]) + mask[NA, :, NA] # 1, nW, 1, wsz*wsz, wsz*wsz
        x = tf.reshape(x, [-1, num_heads, I, I])
        return x

    @shift(shift_size)
    @window_partitioning(window_size)
    def main(x, mask=None):
        _, H, W, C = x.shape
        x = tf.reshape(x, [-1, H*W, C])
        Q, K, V = [Dense(emb_dim)(x) for _ in range(3)]
        Q, K, V = [separate_heads(num_heads)(target) for target in [Q, K, V]]  # B, nH, I, C/nH
        x = tf.matmul(Q, K, transpose_b=True) # B, nH, I, I
        dims = tf.cast(K.shape[-1], tf.float32)
        x = x / tf.math.sqrt(dims) + EPSILON
        if mask != None:
            x += get_relative_position(H, W) # B*nw, nH, w*w, w*w
            x = set_mask(x, mask)
        x = tf.nn.softmax(x, -1)
        x = Dropout(drop_rate)(x)

        x = tf.matmul(x, V)
        x = tf.transpose(x, [0, 2, 1, 3]) # [B, nH, I, C/nH] > [B, I, nH, C/nH]
        x = tf.reshape(x, [-1, H, W, C])
        x = Dense(emb_dim)(x)
        x = Dropout(drop_rate)(x)
        return x
    return main

def swin_encoder(emb_dim, window_size, num_heads, drop_rate=0.2):
    def get_mask(H, W, window_size, shift_size, mask_value=-100.):
        """
        https://www.researchgate.net/publication/358760702/figure/fig4/AS:1126115154366472@1645498183838/Illustration-of-masked-MSA.png
        https://github.com/microsoft/Swin-Transformer/issues/38

        :return: [nW, wsz*wsz, wsz*wsz]
        """
        x = np.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                x[:, h, w, :] = cnt
                cnt += 1
        # window partition
        x = tf.reshape(x, [-1, H // window_size, window_size, W // window_size, window_size])
        x = tf.transpose(x, [0, 1, 3, 2, 4])  # B, H/w, W/w, w, w
        x = tf.reshape(x, [-1, window_size * window_size])
        #
        x = tf.expand_dims(x, 1) - tf.expand_dims(x, 2)
        x = tf.where(x != 0, mask_value, 0.)
        return x

    shift_size = window_size // 2
    def main(input):
        _, H, W, C = input.shape
        x = LayerNormalization(epsilon=EPSILON)(input)
        ### W-MSA
        # if shift_size < 0 make not shift image
        x = wMSA(emb_dim, num_heads, window_size, shift_size=-1)(x, None)
        ### SW-MSA
        mask = get_mask(H, W, window_size, shift_size)
        x = wMSA(emb_dim, num_heads, window_size, shift_size)(x, mask)

        # FFN
        x = Dropout(drop_rate)(x) + input
        x_ = MLP(emb_dim * 4)(x)
        x_ = Dropout(drop_rate)(x_)
        return x + x_
    return main



########################################################################################################################
def separate_heads(num_heads): # B, n_patch, embedding_dims
    def main(x):
        _, I, C = x.shape
        assert x.shape[-1] % num_heads == 0
        x = tf.reshape(x, [-1, I, num_heads, C//num_heads])
        x = tf.transpose(x, perm=[0, 2, 1, 3]) # B, nHeads, I, C/nH
        return x
    return main

def MSA(num_heads, dim, drop_rate=0.2):
    # self-attention possesses a strong inductive bias towards “token uniformity”
    def self_attention(Q, K, V):  # B, nHeads, I, C/nH
        score = tf.matmul(Q, K, transpose_b=True)  # B, n_head, I, I
        dims = tf.cast(K.shape[-1], tf.float32)  # emb_dims
        scaled_score = score / tf.math.sqrt(dims) + EPSILON
        weights = tf.nn.softmax(scaled_score, axis=-1)
        # weights = Dropout(drop_rate)(weights)
        output = tf.matmul(weights, V)  # B, n_heads, I, embedding_dims
        return output, weights

    def main(x):
        _, I, C = x.shape
        x = LayerNormalization(epsilon=EPSILON)(x)
        Q, K, V = [Dense(dim)(x) for _ in range(3)]
        Q, K, V = [separate_heads(num_heads)(target) for target in [Q, K, V]] # B, nHeads, I, C/nH

        x, weights = self_attention(Q, K, V)
        x = tf.transpose(x, [0, 2, 1, 3])  # B, I, nHeads, C/nH
        x = tf.reshape(x, [-1, I, C]) # B, n_patch, embedding_dims
        x = Dense(dim)(x)
        x = Dropout(drop_rate)(x)
        return x
    return main

def encoder(num_heads, embed_dims, drop_rate=0.2):
    def main(inputs):
        x = MSA(num_heads, embed_dims)(inputs) # B, n_patch, emb_dims
        # skip connections play a key role in mitigating rank collapse
        x = Dropout(drop_rate)(x) + inputs # B, n_patch, emb_dims

        x_ = MLP([embed_dims*2, embed_dims])(x)
        x_ = Dropout(drop_rate)(x_) # B, n_patch, embed_dims
        return x + x_
    return main


# MLPs can slow down the convergence by increasing their Lipschitz constant
def MLP(units, drop_rate=0.2):
    units = [units] if type(units) == int else units
    def main(x):
        x = LayerNormalization(epsilon=EPSILON)(x)
        for unit in units:
            x = Dense(unit, 'gelu')(x)
            x = Dropout(drop_rate)(x)
        return x
    return main

def patch_encoding(embed_dims, pos_emb=True):
    def main(x):
        x = Dense(units=embed_dims)(x)
        """
        Improved Robustness of Vision Transformer via PreLayerNorm in Patch Embedding
            https://arxiv.org/abs/2111.08413
        if the scale of operand X increases, the effect of adding Epos would vanish
        """
        x = LayerNormalization()(x)
        if pos_emb == True:
            position_embed = Embedding(input_dim=x.shape[1], output_dim=embed_dims)
            x += position_embed(tf.range(x.shape[1]))
        return x
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
        _, H, W, C = patches.shape
        return tf.reshape(patches, [-1, H * W, C])
    return main