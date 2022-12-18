class hparams:

    train_or_test = 'train'
    output_dir = 'logs/miniseg'
    aug = None
    latest_checkpoint_file = 'checkpoint_latest.pt'
    total_epochs = 100
    epochs_per_checkpoint = 10
    batch_size = 16
    ckpt = None
    init_lr = 0.002
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    debug = False
    mode = '2d' # '2d or '3d'
    in_class = 3
    out_class = 1

    crop_or_pad_size = 256,256,1 # if 2D: 256,256,1
    patch_size = 256,256,1 # if 2D: 128,128,1 
    num_workers = 8

    # for test
    patch_overlap = 4,4,0 # if 2D: 4,4,0

    fold_arch = '*.jpg'

    save_arch = '.nii.gz'

    source_train_dir = 'thyroid_dataset/train/inputs'
    label_train_dir = 'thyroid_dataset/train/gt'
    source_test_dir = 'thyroid_dataset/val/inputs'
    label_test_dir = 'thyroid_dataset/val/gt'


    output_dir_test = 'results/miniseg'