import os
import shutil

class models_genesis_config:
    model = "Unet3D"
    suffix = "genesis_head_mri"
    exp_name = model + "-" + suffix
    sample_path = './sample_images/'
    if not os.path.exists(os.path.join(sample_path, exp_name)):
        os.makedirs(os.path.join(sample_path, exp_name))

    # data
    data = "../generated_cubes_MRI/"
    train_fold = [0] #[1,2,3,4,5]
    valid_fold = [0] #[6,7]
    test_fold = [0] #[8,9]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 32
    input_rows = 64
    input_cols = 64 
    input_deps = 32
    nb_class = 1
    
    # model pre-training
    verbose = 1
    weights = None
    batch_size = 32
    optimizer = "sgd"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = 1000
    patience = 10
    lr = 1

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.8
    flip_rate = 0.4
    
    # logs
    model_path = "./pretrained_weights_MRI"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "logs_MRI")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
