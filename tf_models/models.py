import tensorflow as tf
# class CotNetLayer(tf.keras.Mo)





dim = 3
kernel_size = 5
stride = (2,2)



key_embed = tf.keras.Sequential([
    tf.keras.layers.ZeroPadding2D(padding=kernel_size//2),
    tf.keras.layers.Conv2D(dim, kernel_size, padding="valid", kernel_initializer="he_normal", activation="relu"),
])

conv1x1