import tensorflow as tf
from tensorflow.keras import layers, Model


# ==============================================================================
# =                                  networks                                  =
# ==============================================================================


# 生成器模型： 1x1x128 ==> G ==> 64x64x3
def Generator(input_shape=(1, 1, 128), output_channels=3, output_dim=64, n_upsamplings=4, name='Generator'):
    inputs = layers.Input(shape=input_shape)

    # 1: 1x1x128 ==> 4x4x512
    d = min(output_dim * 2 ** (n_upsamplings - 1), output_dim * 8)
    h = layers.Conv2DTranspose(d, 4, strides=1, padding='valid', use_bias=False)(inputs)
    h = layers.BatchNormalization()(h)
    h = tf.nn.relu(h)

    # 2: upsampling  4x4x512 ==> 8x8x256 ==> 16x16x128 ==> 32x32x64
    for i in range(n_upsamplings - 1):
        d = min(output_dim * 2 ** (n_upsamplings - 2 - i), output_dim * 8)
        h = layers.Conv2DTranspose(d, 4, strides=2, padding='same', use_bias=False)(h)
        h = layers.BatchNormalization()(h)
        h = tf.nn.relu(h)

    # 3: 32x32x64 ==> 64x64x3
    h = layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same')(h)
    h = tf.tanh(h)

    return Model(inputs=inputs, outputs=h, name=name)


# 判别器模型： 64x64x3 ==> D ==> 1x1x1
def Discriminator(input_shape=(64, 64, 3), dim=64, n_downsamplings=4, name='Discriminator'):
    inputs = layers.Input(shape=input_shape)

    # 1: 64x64x3 ==> 32x32x64
    h = layers.Conv2D(dim, 4, strides=2, padding='same')(inputs)
    h = layers.LeakyReLU(alpha=0.2)(h)

    # 2: 32x32x64 ==> 16x16x128 ==> 8x8x256 ==> 4x4x512
    for i in range(n_downsamplings - 1):
        d = min(dim * 2 ** (i + 1), dim * 8)
        h = layers.Conv2D(d, 4, strides=2, padding='same', use_bias=False)(h)
        h = layers.LayerNormalization()(h)  # 判别器不能用batchNorm
        h = layers.LeakyReLU(alpha=0.2)(h)

    # 3: 4x4x512 ==> 1x1x1
    h = layers.Conv2D(1, 4, strides=1, padding='valid')(h)

    return Model(inputs=inputs, outputs=h, name=name)


