from main import main
from config import preset_config
from commandline_config import Config
import os.path as osp

if __name__ == '__main__':
    for model in ["miniseg", "unet", "segnet", "deeplabv3", "pspnet"]:
        config = Config(preset_config=preset_config)
        config.model = model
        config.output_dir = osp.join(config.output_dir, model)
        config.output_dir_test = osp.join(config.output_dir_test, model)
        config.output_dir_seg = osp.join(config.output_dir_seg, model)
        config.train_or_test = "train"
        main(config)
        config.train_or_test = "test"
        main(config)