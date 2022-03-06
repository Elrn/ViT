from modules import *

########################################################################################################################
def Swin(num_classes):
    """
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
        https://arxiv.org/abs/2103.14030

    :return:
    """
    emb_dim = 96
    depths = [1, 1, 3, 1]
    num_heads = [3, 6, 12, 24]
    def main(x):
        for i in range(len(depths)):
            x = patch_partition(emb_dim*(2**i), patch_size=4)(x) if i == 0 else patch_merging()(x)
            for j in range(depths[i]):
                x = swin_encoder(emb_dim*(2**i), window_size=7, num_heads=num_heads[i])(x)

        x = LayerNormalization()(x) # B H W C
        x = tf.reduce_mean(x, [1,2]) # B C
        x = Dense(num_classes, kernel_initializer=TruncatedNormal)(x)
        return x
    return main

########################################################################################################################
def ViT(num_classes, patch_size, num_encodings=8, num_heads=4, embed_dims=64):
    """
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
        https://arxiv.org/abs/2010.11929
    """
    def main(x):
        ### Patch
        x = make_patches(patch_size)(x)
        x = patch_encoding(embed_dims)(x) # B, n_patch, embed_dims
        ### Transformer Encoding
        for _ in range(num_encodings):
            x = encoder(num_heads, embed_dims)(x)  # B, n_patch, embed_dims
        ###
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        # skip connections and MLPs stop the output from degeneration
        x = MLP([2048, 1024], drop_rate=0.5)(x)
        output = Dense(num_classes)(x)
        return output
    return main

########################################################################################################################
