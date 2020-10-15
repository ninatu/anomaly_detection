import os
import json
from PIL import Image

from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


def ensure_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)


class Logger(object):
    def __init__(self, root):
        self.root = root
        self.grid_dir = os.path.join(self.root, 'images')
        self.tensorboard = os.path.join(self.root, 'tensorboard')

        ensure_dir(self.grid_dir)
        ensure_dir(self.tensorboard)

        self.writer = SummaryWriter(self.tensorboard)

    def add_scalar(self, name, value, iter):
        self.writer.add_scalar(name, value, iter)

    def add_scalars(self, main_name, tag_value_dict, iter):
        self.writer.add_scalars(main_name, tag_value_dict, iter)

    def add_histogram(self, name, values, iter):
        self.writer.add_histogram(name, values, iter)

    def save_grid(self, images, grid_size, name, nrow=4):
        n = images.size(0)
        grid = make_grid(images, nrow=nrow)
        grid = (grid * 0.5 + 0.5) * 255
        grid = grid.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        grid = Image.fromarray(grid)
        grid = grid.resize((grid_size * nrow, grid_size * int(n / nrow)), Image.NEAREST)
        grid.save(os.path.join(self.grid_dir, name))

    def save_config(self, config_dict, name):
        with open(os.path.join(self.root, name), 'w') as f_out:
            json.dump(config_dict, f_out)
