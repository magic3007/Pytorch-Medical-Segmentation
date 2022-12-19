class hparams:

    train_or_test = "train"
    model = "unet"
    output_dir = f"logs/{model}"
    aug = None
    latest_checkpoint_file = "checkpoint_latest.pt"
    total_epochs = 5
    epochs_per_checkpoint = 10
    batch_size = 16
    ckpt = None
    init_lr = 0.002
    scheduer_step_size = 2
    scheduer_gamma = 0.8
    debug = False
    random_seed = 114514
    mode = "2d"  # '2d or '3d'
    in_class = 3
    out_class = 1

    crop_or_pad_size = 256, 256, 1  # if 2D: 256,256,1
    patch_size = 256, 256, 1  # if 2D: 128,128,1
    num_workers = 8
    queue_length = 100
    samples_per_volume = 5

    # for test
    patch_overlap = 4, 4, 0  # if 2D: 4,4,0

    fold_arch = "*.png"

    save_arch = ".nii.gz"

    source_train_dir = "thyroid_dataset/train/inputs"
    label_train_dir = "thyroid_dataset/train/gt"
    source_test_dir = "thyroid_dataset/val/inputs"
    label_test_dir = "thyroid_dataset/val/gt"

    output_dir_test = "results/miniseg"
