import os
import shutil

class models_genesis_config:
    model = "Unet3D"
    suffix = "genesis_head_ct"
    exp_name = model + "-" + suffix
    sample_path = './sample_images_ct/'

    if not os.path.exists(os.path.join(sample_path, exp_name)):
        os.makedirs(os.path.join(sample_path, exp_name))
    # data
    data = "../generated_cubes_CT/"
    train_fold=[1, 2, 3, 4, 5, 6] #
    valid_fold=[7, 8] #
    test_fold=[9, 10] #
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 32
    input_rows = 64
    input_cols = 64 
    input_deps = 32
    nb_class = 1
    
    # model pre-training
    verbose = 1
    weights = "./pretrained_weights_CT/Genesis_Head_CT.pt"
    batch_size = 6
    optimizer = "sgd"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = 1000
    patience = 5
    lr = 1

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    # logs
    model_path = "./pretrained_weights_CT"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "logs_CT")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
