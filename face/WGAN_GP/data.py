# 准备好dataset，其中数据集中的每张图片都经过标准化处理为64*64的大小

import tensorflow as tf
from face.WGAN_GP.utils import dataset_utils


# ==============================================================================
# =                                  datasets                                  =
# ==============================================================================

def make_celeba_dataset(img_paths, batch_size, resize=64, drop_remainder=True, shuffle=True, repeat=1):
    @tf.function
    def _map_fn(img):
        crop_size = 108
        img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
        img = tf.image.resize(img, [resize, resize])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5 - 1
        return img

    dataset = dataset_utils.disk_image_batch_dataset(img_paths,
                                                     batch_size,
                                                     drop_remainder=drop_remainder,
                                                     map_fn=_map_fn,
                                                     shuffle=shuffle,
                                                     repeat=repeat)
    img_shape = (resize, resize, 3)
    len_dataset = len(img_paths) // batch_size

    return dataset, img_shape, len_dataset











