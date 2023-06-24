import math
import tensorflow as tf

class CotNetLayer(tf.keras.layers.Layer):
    def __init__(self, dim, kernel_size, radix=2, factor=2, reduction_factor=4, share_planes=8):
        super(CotNetLayer, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.radix = radix
        self.attn_chs = max(dim * radix // reduction_factor, 32)
        self.act_func = tf.keras.activations.swish

        # ======================
        #         LAYERS
        # ======================
        self.key_embed = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(padding=kernel_size//2),
            tf.keras.layers.Conv2D(dim, kernel_size, padding="valid"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])

        self.theta_sigma_embed = tf.keras.Sequential([
            # theta weight layers
            tf.keras.layers.Conv2D(dim // factor, 1, padding="valid", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            # sigma weight layers
            tf.keras.layers.Conv2D(math.pow(kernel_size, 2) * dim // share_planes, kernel_size=1, padding="valid"),
            tf.keras.layers.GroupNormalization(dim // share_planes, epsilon=1e-5)
        ])

        self.val_embed = tf.keras.Sequential([
            tf.keras.layers.Conv2D(dim, kernel_size=1, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(self.act_func)
        ])

        self.squeeze_excitation = tf.keras.Sequential([
            tf.keras.layers.Conv2D(attn_chs, 1, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(radix*dim, 1, padding="same"),
        ])

        # TODO: due to CUDA dependency could not be implemented
        # self.local_conv = LocalConvolution()
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x: tf.Tensor):
        k = self.key_embed(x)
        qk = tf.concat([x, k], axis=-1)
        b, qk_hh, qk_ww, c = qk.shape

        w = self.theta_sigma_embed(qk)
        w = tf.reshape(b, 1, -1, math.pow(self.kernel_size, 2), qk_hh, qk_ww)

        x = self.val_embed(x)
        # x = self.local_conv(x)
        # x = self.bn(x)
        # x = self.act_func(x)

        B, C, H, W = x.shape
        x = tf.reshape(x, [B, C, 1, H, W])
        k = tf.reshape(k, [B, C, 1, H, W])
        x = tf.concat([x, k], axis=2)

        x_gap = tf.reduce_sum(x, axis=2)
        x_gap = tf.reduce_mean(x_gap, axis=(2,3), keepdims=True)
        x_gap = tf.transpose(x_gap, perm=[0, 2, 3, 1])
        x_attn = self.squeeze_excitation(x_gap)
        x_attn = tf.reshape(x_attn, [B, C, self.radix])
        x_attn = tf.keras.activations.softmax(x_attn)

        out = tf.reduce_sum(tf.matmul(x, tf.reshape(x_attn, [B, C, self.radix, 1, 1])), axis=2)
        out = tf.transpose(out, [0, 2, 3, 1])
        return out
