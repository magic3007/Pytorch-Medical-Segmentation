#%%
import os
import os.path as osp
from commandline_config import Config
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torchvision import transforms
import torchio
from torchio.transforms import ZNormalization
from tqdm import tqdm
from utils.metric import metric
from config import preset_config
from gen_video import gen_video

#%%


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


#%%


def train(config):

    from data_function import MedData_train

    os.makedirs(config.output_dir, exist_ok=True)

    if config.mode == "2d":

        if config.model == "unet":
            from models.two_d.unet import Unet

            model = Unet(in_channels=config.in_class, classes=config.out_class + 1)
        elif config.model == "miniseg":
            from models.two_d.miniseg import MiniSeg

            model = MiniSeg(in_input=config.in_class, classes=config.out_class + 1)
        elif config.model == "fcn":
            from models.two_d.fcn import FCN32s as fcn

            model = fcn(in_class=config.in_class, n_class=config.out_class + 1)
        elif config.model == "segnet":
            from models.two_d.segnet import SegNet

            model = SegNet(input_nbr=config.in_class, label_nbr=config.out_class + 1)
        elif config.model == "deeplabv3":
            from models.two_d.deeplab import DeepLabV3

            model = DeepLabV3(in_class=config.in_class, class_num=config.out_class + 1)
        elif config.model == "restnet34unetplus":
            from models.two_d.unetpp import ResNet34UnetPlus

            model = ResNet34UnetPlus(
                num_channels=config.in_class, num_class=config.out_class + 1
            )
        elif config.model == "pspnet":
            from models.two_d.pspnet import PSPNet

            model = PSPNet(in_class=config.in_class, n_classes=config.out_class + 1)
        else:
            assert False, "model not found"

    elif config.mode == "3d":
        # from models.three_d.unet3d import UNet3D
        # model = UNet3D(in_channels=config.in_class, out_channels=config.out_class+1, init_features=32)

        from models.three_d.residual_unet3d import UNet

        model = UNet(
            in_channels=config.in_class, n_classes=config.out_class + 1, base_n_filter=2
        )

        # from models.three_d.fcn3d import FCN_Net
        # model = FCN_Net(in_channels =config.in_class,n_class =config.out_class+1)

        # from models.three_d.highresnet import HighRes3DNet
        # model = HighRes3DNet(in_channels=config.in_class,out_channels=config.out_class+1)

        # from models.three_d.densenet3d import SkipDenseNet3D
        # model = SkipDenseNet3D(in_channels=config.in_class, classes=config.out_class+1)

        # from models.three_d.densevoxelnet3d import DenseVoxelNet
        # model = DenseVoxelNet(in_channels=config.in_class, classes=config.out_class+1)

        # from models.three_d.vnet3d import VNet
        # model = VNet(in_channels=config.in_class, classes=config.out_class+1)

        # from models.three_d.unetr import UNETR
        # model = UNETR(img_shape=(config.crop_or_pad_size), input_dim=config.in_class, output_dim=config.out_class+1)

    else:
        assert False, "mode not found"

    optimizer = torch.optim.Adam(model.parameters(), lr=config.init_lr)

    scheduler = StepLR(
        optimizer, step_size=config.scheduer_step_size, gamma=config.scheduer_gamma
    )
    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)

    if config.ckpt is not None:
        print("load model:", config.ckpt)
        ckpt = torch.load(config.ckpt, map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(config.device)

        scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    model = model.to(config.device)

    from loss_function import Binary_Loss, DiceLoss
    from torch.nn.modules.loss import CrossEntropyLoss

    criterion_dice = DiceLoss(2).to(config.device)
    criterion_ce = CrossEntropyLoss().to(config.device)

    writer = SummaryWriter(comment=config.model)
    for k in sorted(config.preset_config.keys()):
        writer.add_text(k, str(config[k]))
    
    train_dataset = MedData_train(
        config, config.source_train_dir, config.label_train_dir
    )
    train_loader = DataLoader(
        train_dataset.queue_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    model.train()

    epochs = config.total_epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)

    epoch_bar = tqdm(range(1, epochs + 1))
    for epoch in epoch_bar:
        epoch_bar.set_description(
            f"Epoch {epoch}/{epochs} lr {optimizer.param_groups[0]['lr']:.6f}"
        )
        epoch += elapsed_epochs

        num_iters = 0
        total_dice = 0

        batch_bar = tqdm(enumerate(train_loader))
        for i, batch in batch_bar:

            if config.debug:
                if i > 0:
                    break

            optimizer.zero_grad()

            x = batch["source"]["data"]
            y = batch["label"]["data"]

            y_back = torch.zeros_like(y)
            y_back[(y == 0)] = 1

            x = x.type(torch.FloatTensor).to(config.device)
            y = torch.cat((y_back, y), 1)
            y = y.type(torch.FloatTensor).to(config.device)

            outputs = model(x.squeeze(4))
            outputs = outputs.unsqueeze(4)

            # for metrics
            labels = outputs.argmax(dim=1)
            model_output_one_hot = torch.nn.functional.one_hot(
                labels, num_classes=config.out_class + 1
            ).permute(0, 4, 1, 2, 3)

            loss = criterion_ce(outputs, y.argmax(dim=1)) + criterion_dice(
                outputs, y.argmax(dim=1)
            )

            num_iters += 1
            loss.backward()

            optimizer.step()
            iteration += 1

            y_argmax = y.argmax(dim=1)
            y_one_hot = torch.nn.functional.one_hot(
                y_argmax, num_classes=config.out_class + 1
            ).permute(0, 4, 1, 2, 3)

            (
                false_positive_rate,
                false_negtive_rate,
                dice,
                jaccard,
                precision,
                recall,
                tp,
                fp,
                fn,
                tn,
            ) = metric(
                y_one_hot[:, 1:, :, :].cpu(), model_output_one_hot[:, 1:, :, :].cpu()
            )
            total_dice = total_dice + dice

            ## log
            writer.add_scalar("Training/Loss", loss.item(), iteration)
            writer.add_scalar(
                "Training/false_positive_rate", false_positive_rate, iteration
            )
            writer.add_scalar(
                "Training/false_negtive_rate", false_negtive_rate, iteration
            )
            writer.add_scalar("Training/dice", dice, iteration)

            batch_bar.set_description(
                f"Batch: {i}/{len(train_loader)} loss: {loss.item():.4f} dice: {total_dice / (i+1):.4f}"
            )

        scheduler.step()
        # Store latest checkpoint in each epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            os.path.join(config.output_dir, config.latest_checkpoint_file),
        )

        # Save checkpoint
        if epoch % config.epochs_per_checkpoint == 0:

            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(config.output_dir, f"checkpoint_{epoch:04d}.pt"),
            )

            with torch.no_grad():
                x = x[0].cpu().detach().numpy()
                y = y[0].cpu().detach().numpy()
                outputs = outputs[0].cpu().detach().numpy()
                model_output_one_hot = (
                    model_output_one_hot[0].float().cpu().detach().numpy()
                )
                affine = batch["source"]["affine"][0].numpy()

                y = np.expand_dims(y, axis=1)
                outputs = np.expand_dims(outputs, axis=1)
                model_output_one_hot = np.expand_dims(model_output_one_hot, axis=1)

                source_image = torchio.ScalarImage(tensor=x, affine=affine)
                source_image.save(
                    os.path.join(
                        config.output_dir, f"step-{epoch:04d}-source" + config.save_arch
                    )
                )

                label_image = torchio.ScalarImage(tensor=y[1], affine=affine)
                label_image.save(
                    os.path.join(
                        config.output_dir, f"step-{epoch:04d}-gt" + config.save_arch
                    )
                )

                output_image = torchio.ScalarImage(
                    tensor=model_output_one_hot[1], affine=affine
                )
                output_image.save(
                    os.path.join(
                        config.output_dir,
                        f"step-{epoch:04d}-predict" + config.save_arch,
                    )
                )

    writer.close()


def test(config):

    from data_function import MedData_test

    os.makedirs(config.output_dir_test, exist_ok=True)

    if config.mode == "2d":

        if config.model == "unet":
            from models.two_d.unet import Unet

            model = Unet(in_channels=config.in_class, classes=config.out_class + 1)
        elif config.model == "miniseg":
            from models.two_d.miniseg import MiniSeg

            model = MiniSeg(in_input=config.in_class, classes=config.out_class + 1)
        elif config.model == "fcn":
            from models.two_d.fcn import FCN32s as fcn

            model = fcn(in_class=config.in_class, n_class=config.out_class + 1)
        elif config.model == "segnet":
            from models.two_d.segnet import SegNet

            model = SegNet(input_nbr=config.in_class, label_nbr=config.out_class + 1)
        elif config.model == "deeplabv3":
            from models.two_d.deeplab import DeepLabV3

            model = DeepLabV3(in_class=config.in_class, class_num=config.out_class + 1)
        elif config.model == "restnet34unetplus":
            from models.two_d.unetpp import ResNet34UnetPlus

            model = ResNet34UnetPlus(
                num_channels=config.in_class, num_class=config.out_class + 1
            )
        elif config.model == "pspnet":
            from models.two_d.pspnet import PSPNet

            model = PSPNet(in_class=config.in_class, n_classes=config.out_class + 1)
        else:
            assert False, "model not found"

    elif config.mode == "3d":
        # from models.three_d.unet3d import UNet3D
        # model = UNet3D(in_channels=config.in_class, out_channels=config.out_class+1, init_features=32)

        from models.three_d.residual_unet3d import UNet

        model = UNet(
            in_channels=config.in_class, n_classes=config.out_class + 1, base_n_filter=2
        )

        # from models.three_d.fcn3d import FCN_Net
        # model = FCN_Net(in_channels =config.in_class,n_class =config.out_class+1)

        # from models.three_d.highresnet import HighRes3DNet
        # model = HighRes3DNet(in_channels=config.in_class,out_channels=config.out_class+1)

        # from models.three_d.densenet3d import SkipDenseNet3D
        # model = SkipDenseNet3D(in_channels=config.in_class, classes=config.out_class+1)

        # from models.three_d.densevoxelnet3d import DenseVoxelNet
        # model = DenseVoxelNet(in_channels=config.in_class, classes=config.out_class+1)

        # from models.three_d.vnet3d import VNet
        # model = VNet(in_channels=config.in_class, classes=config.out_class+1)

        # from models.three_d.unetr import UNETR
        # model = UNETR(img_shape=(config.crop_or_pad_size), input_dim=config.in_class, output_dim=config.out_class+1)

    if config.ckpt is not None:
        ckpt_path = config.ckpt
    else:
        ckpt_path = os.path.join(config.output_dir, config.latest_checkpoint_file)
    print("load model:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt["model"])
    model.to(config.device)

    test_dataset = MedData_test(config, config.source_test_dir, config.label_test_dir)
    tran = transforms.Compose(
        [
            ZNormalization(),
        ]
    )
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    elapsed_time = 0
    pbar = tqdm(enumerate(test_dataset.subjects))
    for i, subj in pbar:
        subj = tran(subj)
        grid_sampler = torchio.inference.GridSampler(
            subj,
            config.patch_size,
            config.patch_overlap,
        )

        patch_loader = torch.utils.data.DataLoader(
            grid_sampler, batch_size=config.batch_size
        )
        aggregator = torchio.inference.GridAggregator(grid_sampler)
        model.eval()
        tt = time.time()
        with torch.no_grad():
            for patches_batch in patch_loader:

                input_tensor = patches_batch["source"][torchio.DATA].to(config.device)
                locations = patches_batch[torchio.LOCATION]

                input_tensor = input_tensor.squeeze(4)
                outputs = model(input_tensor)

                outputs = outputs.unsqueeze(4)

                labels = outputs.argmax(dim=1)
                aggregator.add_batch(labels.unsqueeze(1), locations)
        # output_tensor = aggregator.get_output_tensor()
        output_tensor = aggregator.get_output_tensor()
        elapsed_time += time.time() - tt

        gt = subj["label"]["data"]
        (
            false_positive_rate,
            false_negtive_rate,
            dice,
            jaccard,
            precision,
            recall,
            tp,
            fp,
            fn,
            tn,
        ) = metric(gt.cpu(), output_tensor.cpu())
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        affine = subj["source"]["affine"]
        output_image = torchio.ScalarImage(tensor=output_tensor.numpy(), affine=affine)
        output_image.save(
            os.path.join(
                config.output_dir_test, f"{i:04d}-result_int" + config.save_arch
            )
        )

        pbar.set_description(f"Test {i:04d}/ {len(test_dataset.subjects)}")

    dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8)
    print(f"Test dice: {dice:.4f} Test elapsed time: {elapsed_time:.2f}")
    print("total_tp", total_tp)
    print("total_fp", total_fp)
    print("total_fn", total_fn)
    print("total_tn", total_tn)
    print("elapsed_time", elapsed_time)
    gen_video(config.source_test_dir, config.output_dir_test, config.output_dir_seg)


def main(config):
    print(config)
    seed_everything(config.random_seed)
    if config.train_or_test == "train":
        train(config)
    elif config.train_or_test == "test":
        test(config)
    else:
        assert False, "train_or_test must be train or test"

if __name__ == "__main__":
    config = Config(preset_config=preset_config)
    model = config.model
    config.output_dir = osp.join(config.output_dir, model)
    config.output_dir_test = osp.join(config.output_dir_test, model)
    config.output_dir_seg = osp.join(config.output_dir_seg, model)
    main(config) 