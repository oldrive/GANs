import tensorflow as tf
import config
from face.WGAN_GP.utils import image_utils, path_utils
import make_gif
import os

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)  # 设置GPU显存用量按需使用


# ==============================================================================
# =                                  generate images                           =
# ==============================================================================
def generate_images():
    G = tf.keras.models.load_model('model/Generator')

    z = tf.random.normal((100, 1, 1, config.Z_DIM))
    x_fake = G(z, training=False)

    output_dir = os.path.join('output', 'celeba_wgan_gp')
    path_utils.mkdir(output_dir)
    sample_dir = os.path.join(output_dir, 'samples_final')  # 训练过程中生成的图片目录
    path_utils.mkdir(sample_dir)
    img = image_utils.immerge(x_fake, n_rows=10).squeeze()
    image_utils.imwrite(img, os.path.join(sample_dir, 'final-generated-images.jpg'))


# generate_images()


# ==============================================================================
# =                                  make gif                                  =
# ==============================================================================
img_dir = 'output/celeba_wgan_gp/samples_training'
save_dir = 'output/celeba_wgan_gp/training_gif'
path_utils.mkdir(save_dir)
save_path = os.path.join(save_dir, 'celeba_wgan_gp.gif')
make_gif.make_gif(img_dir=img_dir, save_path=save_path)













