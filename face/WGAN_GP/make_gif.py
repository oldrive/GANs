import imageio
from face.WGAN_GP.utils import path_utils


def make_gif(img_dir, save_path, max_frames=0):
    # 将img_dir下的所有图片制作成git并保存在save_path下
    with imageio.get_writer(save_path, mode='I', fps=8) as writer:
        filenames = sorted(path_utils.glob(img_dir, '*.jpg'))
        if max_frames:
            step = len(filenames) // max_frames
        else:
            step = 1
        last = -1
        for i, filename in enumerate(filenames[::step]):
            frame = 2 * (i ** 0.3)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
























