#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torchsummary import summary
import sys
from utils import *
import unet3d
from config_MRI_classification import models_genesis_config
from tqdm import tqdm

print("torch = {}".format(torch.__version__))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

conf = models_genesis_config()
conf.display()

class TargetNet(nn.Module):
    def __init__(self, base_model, n_class=1):
        super(TargetNet, self).__init__()
        self.base_model = base_model
        self.dense_1 = nn.Linear(512, 1024, bias=True)
        self.dense_2 = nn.Linear(1024, n_class, bias=True)

    def forward(self, x):
        self.base_model(x)
        self.base_out = self.base_model.out512
        self.out_glb_avg_pool_ = F.avg_pool3d(self.base_out, kernel_size= self.base_out.size()[2:])
        self.out_glb_avg_pool = self.out_glb_avg_pool_.view(self.base_out.size()[0], -1)
        self.linear_out = self.dense_1(self.out_glb_avg_pool)
        final_out = self.dense_2(F.relu(self.linear_out))
        return final_out

def compute_accuracy(y_pred, y_true):


    y_pred = y_pred.view(y_true.size()[0])
    y_pred = y_pred >= 0.5
    y_true = y_true >= 0.5
    correct =  acc = y_pred.eq(y_true).float().sum()
    accuracy = correct/y_true.size()[0]
    accuracy = accuracy.item()
    return accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train = []
y_train = []
for i, fold in enumerate(tqdm(conf.train_fold)):
    x_file_name = "batch_classification_MRI_labels_"+str(conf.scale)+"_"+str(conf.input_rows)+"x"+str(conf.input_cols)+"x"+str(conf.input_deps)+"_"+str(fold)+".npy"
    labels_file_name = "batch_classification_MRI_true_labels_" + str(conf.scale) + "_" + str(conf.input_rows) + "x" + str(conf.input_cols) + "x" + str(conf.input_deps) + "_" + str(fold) + ".npy"
    s = np.load(os.path.join(conf.data, x_file_name))
    x_train.extend(s)
    labels = np.load(os.path.join(conf.data, labels_file_name))
    y_train.extend(labels)

x_train = np.expand_dims(np.array(x_train), axis=1)

'''
x_valid = []
y_valid = []
for i, fold in enumerate(tqdm(conf.valid_fold)):
    x_file_name = "batch_CT_" + str(conf.scale) + "_" + str(conf.input_rows) + "x" + str(conf.input_cols) + "x" + str(conf.input_deps) + "_" + str(fold) + ".npy"
    labels_file_name = "batch_CT_labels_" + str(conf.scale) + "_" + str(conf.input_rows) + "x" + str(conf.input_cols) + "x" + str(conf.input_deps) + "_" + str(fold) + ".npy"
    s = np.load(os.path.join(conf.data, x_file_name))
    x_valid.extend(s)
    y_valid.extend(os.path.join(conf.data, labels_file_name))
'''

x_test = []
y_test = []
for i, fold in enumerate(tqdm(conf.test_fold)):
    x_file_name = "batch_classification_MRI_labels_"+str(conf.scale)+"_"+str(conf.input_rows)+"x"+str(conf.input_cols)+"x"+str(conf.input_deps)+"_"+str(fold)+".npy"
    labels_file_name = "batch_classification_MRI_true_labels_" + str(conf.scale) + "_" + str(conf.input_rows) + "x" + str(conf.input_cols) + "x" + str(conf.input_deps) + "_" + str(fold) + ".npy"
    s = np.load(os.path.join(conf.data, x_file_name))
    x_test.extend(s)
    labels = np.load(os.path.join(conf.data, labels_file_name))
    y_test.extend(labels)

x_test = np.expand_dims(np.array(x_test), axis=1)

x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

print (x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# prepare the 3D model        
base_model = unet3d.UNet3D()

if conf.type_of_model == 'MG_pretrained':
    #Load pre-trained weights
    weight_dir = './pretrained_weights_MRI/Genesis_Head_MRI.pt'
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['state_dict']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]

    base_model.load_state_dict(unParalled_state_dict)

target_model = TargetNet(base_model)
target_model.to(device)
target_model = nn.DataParallel(target_model, device_ids = [i for i in range(torch.cuda.device_count())])
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(target_model.parameters(), conf.lr, momentum=0.9, weight_decay=0.0, nesterov=False)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(conf.patience * 0.8), gamma=0.5)


# to track the training loss as the model trains
train_losses = []
train_accuracies = []
# to track the validation loss as the model trains
test_losses = []
test_accuracies = []

avg_train_losses = []
avg_test_losses = []

avg_train_accuracies = []
avg_test_accuracies = []


best_loss = 100000
intial_epoch =0
num_epoch_no_improvement = 0

# train the model
for epoch in range(intial_epoch, conf.nb_epoch):
    scheduler.step(epoch)
    target_model.train()
    index = [i for i in range(x_train.shape[0])]
    random.shuffle(index)
    x_ = copy.deepcopy(x_train[index])
    y_ = copy.deepcopy(y_train[index])


    for iteration, bi in enumerate(range(0, y_.shape[0] - conf.batch_size, conf.batch_size)):
        x = x_[bi:bi + conf.batch_size]
        y = y_[bi:bi + conf.batch_size]
        x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device)
        pred = F.sigmoid(target_model(x))
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(round(loss.item(), 2))
        train_accuracies.append(compute_accuracy(pred, y))
        

        if (iteration + 1) % 5 ==0:
            print('Epoch [{}/{}], iteration {}, Loss: {:.6f},  Accuracy: {:.6f}'
                .format(epoch + 1, conf.nb_epoch, iteration + 1, np.average(train_losses), np.average(train_accuracies)))
            sys.stdout.flush()



    '''
    with torch.no_grad():
        target_model.eval()
        print("validating....")
        for i in range(int(x_valid.shape[0]//conf.batch_size)):
            x, y = next(validation_generator)
            x, y = x.float().to(device), y.float().to(device)
            pred = F.sigmoid(target_model(x))
            loss = criterion(pred, y)
    '''

    with torch.no_grad():
        target_model.eval()
        print("test data....")
        index = [i for i in range(x_test.shape[0])]
        random.shuffle(index)
        x_ = copy.deepcopy(x_test[index])
        y_ = copy.deepcopy(y_test[index])

        for iteration, bi in enumerate(range(0, y_.shape[0] - conf.batch_size, conf.batch_size)):
            x = x_[bi:bi + conf.batch_size]
            y = y_[bi:bi + conf.batch_size]
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device)
            pred = F.sigmoid(target_model(x))
            loss = criterion(pred, y)
            test_losses.append(round(loss.item(), 2))
            test_accuracies.append(compute_accuracy(pred, y))

    train_loss = np.average(train_losses)
    test_loss = np.average(test_losses)
    avg_train_losses.append(train_loss)
    avg_test_losses.append(test_loss)
    

    train_accuracy = np.average(train_accuracies)
    test_accuracy = np.average(test_accuracies)
    avg_train_accuracies.append(train_accuracy)
    avg_test_accuracies.append(test_accuracy)
    
    print("Epoch {}, test loss is {:.4f}, training loss is {:.4f}".format(epoch+1, test_loss, train_loss))
    print("Epoch {}, test accuracy is {:.4f}, training accuracy is {:.4f}".format(epoch+1, test_accuracy, train_accuracy))
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    if test_loss < best_loss:
        print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, test_loss))
        best_loss = test_loss
        num_epoch_no_improvement = 0
        #save model
        torch.save({
            'epoch': epoch + 1,
            'state_dict' : target_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },os.path.join(conf.model_path, 'Genesis_Head_MRI_' + str(conf.type_of_model) +'_classification.pt'))
        print("Saving model ", os.path.join(conf.model_path, 'Genesis_Head_MRI_'+ str(conf.type_of_model) +'_classification.pt'))
    
    else:
        print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss, num_epoch_no_improvement))
        num_epoch_no_improvement += 1
    
    if num_epoch_no_improvement == conf.patience:
        print("Early Stopping")
        break
    
    with open(os.path.join(conf.model_path, 'Genesis_Head_MRI_classification_'+ str(conf.type_of_model) +'_logs.csv'), 'a') as out_stream:
        out_stream.write(str(epoch) + ', ' + str(train_accuracy) + ', ' + str(test_accuracy) + ', ' + str(train_loss) + ', ' + str(test_loss) + '\n')

    sys.stdout.flush()