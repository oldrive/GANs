import functools

import tensorflow as tf
import tqdm
import os

import data
import module
import loss
import config
from face.WGAN_GP.utils import path_utils, image_utils


tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)  # 设置GPU显存用量按需使用
# ==============================================================================
# =                               0.   output_dir                              =
# ==============================================================================
output_dir = os.path.join('output', 'celeba_wgan_gp')
path_utils.mkdir(output_dir)
sample_dir = os.path.join(output_dir, 'samples_training')  # 训练过程中生成的图片目录
path_utils.mkdir(sample_dir)


# ==============================================================================
# =                               1.   data                                    =
# ==============================================================================
img_paths = path_utils.glob('data/img_align_celeba', '*.png')  # 得到所有图片的相对路径
# print(img_paths)
dataset, shape, len_dataset = data.make_celeba_dataset(img_paths, config.BATCH_SIZE)  # len_dataset是指batch的数量


# ==============================================================================
# =                               2.   model                                   =
# ==============================================================================
G = module.Generator(input_shape=(1, 1, config.Z_DIM), n_upsamplings=config.N_G_UPSAMPLINGS)
D = module.Discriminator(input_shape=shape, n_downsamplings=config.N_D_DOWNSAMPLINGS)
d_loss_fn, g_loss_fn = loss.get_wgan_loss_fn()
G_optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, beta_1=config.BETA_1)
D_optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, beta_1=config.BETA_1)


# ==============================================================================
# =                               3.   train_step                              =
# ==============================================================================
@tf.function
def train_step_G():
    with tf.GradientTape() as tape:
        z = tf.random.normal(shape=(config.BATCH_SIZE, 1, 1, config.Z_DIM))
        x_fake = G(z, training=True)
        fake_logit = D(x_fake, training=True)
        G_loss = g_loss_fn(fake_logit)
    G_gradient = tape.gradient(G_loss, G.trainable_variables)
    G_optimizer.apply_gradients(zip(G_gradient, G.trainable_variables))

    return G_loss


@tf.function
def train_step_D(x_real):
    with tf.GradientTape() as tape:
        z = tf.random.normal(shape=(config.BATCH_SIZE, 1, 1, config.Z_DIM))
        x_fake = G(z, training=True)
        fake_logit = D(x_fake, training=True)
        real_logit = D(x_real, training=True)
        wgan_d_loss = d_loss_fn(real_logit, fake_logit)
        gp = loss.gradient_penalty(functools.partial(D, training=True), x_real, x_fake)
        D_loss = wgan_d_loss + config.GP_WEIGHT * gp
    D_gradient = tape.gradient(D_loss, D.trainable_variables)
    D_optimizer.apply_gradients(zip(D_gradient, D.trainable_variables))

    return D_loss


# ==============================================================================
# =                               4.   train                                   =
# ==============================================================================


z = tf.random.normal((100, 1, 1, config.Z_DIM))  # 固定值的噪声码，用来判断使用当前生成器生成图片的质量


def train():
    for epoch in tqdm.trange(config.EPOCHS, desc='Epoch Loop'):
        for x_real in tqdm.tqdm(dataset, desc='Batch Loop', total=len_dataset):
            D_loss = train_step_D(x_real)

            if D_optimizer.iterations.numpy() % config.N_D == 0:  # 判别器每训练了N_D次后，才训练一次生成器
                G_loss = train_step_G()

            if G_optimizer.iterations.numpy() % 100 == 0:  #生成器每训练100次，生成图片并保存到指定目录下
                x_fake = G(z, training=False)
                img = image_utils.immerge(x_fake, n_rows=10).squeeze()
                image_utils.imwrite(img, os.path.join(sample_dir, 'epoch-%09d-iter-%09d.jpg' % (epoch + 1, G_optimizer.iterations.numpy())))

        print('epoch:%d, g_loss:%f, d_loss:%f' % (epoch + 1, G_loss, D_loss))


# ==============================================================================
# =                               5.   save model                            =
# ==============================================================================


def save_model(model):
    model.save('./model/Generator', save_format="tf")


# ==============================================================================
# =                               6.   run                                     =
# ==============================================================================


train()
save_model(G)






