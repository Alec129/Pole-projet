import os
import torch

num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 40))
torch.set_num_threads(num_cpus)
torch.set_num_interop_threads(num_cpus)
print(f">>> Configuration Multi-coeurs : {num_cpus} threads utilises.")

import numpy as np
import matplotlib.pyplot as plt
import scipy


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

from model import *



categories = [
    'Healthy',
    'Motor_1_Stuck',
    'Motor_1_Steady_state_error',
    'Motor_2_Stuck',
    'Motor_2_Steady_state_error',
    'Motor_3_Stuck',
    'Motor_3_Steady_state_error',
    'Motor_4_Stuck',
    'Motor_4_Steady_state_error'
]


def import_data(link : str ,link_test : str):
    """import the data from the dataset

    Args:
        link (str): _description_
        link_test (str): _description_

    Returns:
        data_X : train data
        data_Y : train class
        data_test_X : test data
        data_test_Y : test class
    """
    dic = {'Healthy' : 0, 
           'Motor_1_Stuck' : 1,
           'Motor_1_Steady_state_error' : 2,
           'Motor_2_Stuck' : 3,
           'Motor_2_Steady_state_error' : 4,
           'Motor_3_Stuck' : 5,
           'Motor_3_Steady_state_error' : 6,
           'Motor_4_Stuck' : 7,
           'Motor_4_Steady_state_error' : 8
    }


    mat = scipy.io.loadmat(link)
    mat_test = scipy.io.loadmat(link_test)

    X_sim = np.array(mat['X_array'][0])
    data_X = np.array([X_sim[i] for i in range(len(X_sim))])
    Y_sim = np.array(mat['y_array'][0])
    
    data_Y = np.array([dic[Y_sim[i][0]] for i in range(len(Y_sim))])

    X_real = np.array(mat_test['X_test_array'][0])
    data_test_X = np.array([X_real[i] for i in range(len(X_real))])
    Y_real = np.array(mat_test['y_test_array'][0])
    data_test_Y = np.array([dic[Y_real[i][0]]for i in range(len(Y_real))])
    
    return data_X, data_Y, data_test_X, data_test_Y


def import_data_2():
    # import data for the augmented dataset
    data_X = np.zeros((3600,1000,6))
    data_Y = np.zeros(3600)
    
    for i in [0,1,3,5,7]:
        l = [0,1,3,5,7].index(i)
        for j in range(400):
            if i == 0:
                link = '../mydataset/Healthy/dataset_' + str(j + 1) + '.mat'
                mat = scipy.io.loadmat(link)
                data_X[i*400 + j,:,:3] = np.array(mat['trajCmds'])
                data_X[i*400 + j,:,3:6] = np.array(mat['trajResps'])
                data_Y[i*400 + j] = i

            else : 
                link_1 = '../mydataset/Motor_' + str(l) + '_Steady_state_error/dataset_' + str(j + 1) + '.mat'
                link_2 = '..mydataset/Motor_' + str(l) + '_Stuck/dataset_' + str(j + 1) + '.mat'
                mat = scipy.io.loadmat(link_1)
                data_X[i*400 + j,:,:3] = np.array(mat['trajCmds'])
                data_X[i*400 + j,:,3:6] = np.array(mat['trajResps'])
                data_Y[i*400 + j] = i

                mat_2 = scipy.io.loadmat(link_2)
                data_X[(i+1)*400 + j,:,:3] = np.array(mat_2['trajCmds'])
                data_X[(i+1)*400 + j,:,3:6] = np.array(mat_2['trajResps'])
                data_Y[(i+1)*400 + j] = i + 1


    return data_X, data_Y

link = '../mydataset/my_dataset_train.mat'
link_test = '../mydataset/my_dataset_test.mat'
# Data set initial 
X_sim, Y_sim, X_real, Y_real = import_data(link,link_test)

## Data set augmente
#X_sim_1, Y_sim_1, X_real, Y_real = import_data(link,link_test)
#X_sim_2, Y_sim_2 = import_data_2()

#X_sim = np.concatenate((X_sim_1,X_sim_2))
#Y_sim = np.concatenate((Y_sim_1,Y_sim_2))

def normalize(data : np.ndarray) -> np.ndarray:
    """normalize data 

    Args:
        data (np.ndarray): input data

    Returns:
        np.ndarray: normalize data
    """
    # normalization of the data for a trajectory
    mean = data.mean(axis=(0,1))
    std = data.std(axis=(0,1)) + 1e-8

    return mean,std

def treat_data(X_sim,X_real):
    # add the residual to vector (residual = trajectory without error - trajectory with error) and the derivated of the residual
    dim_sim = np.shape(X_sim)
    dim_real = np.shape(X_real)

    X_sim_residual = np.zeros((dim_sim[0],dim_sim[1],dim_sim[2] + 9))
    X_real_residual = np.zeros((dim_real[0],dim_real[1],dim_real[2] + 9))


    X_sim_residual[:,:,:6] = X_sim
    X_sim_residual[:,:,6:9] = X_sim[:,:,:3] - X_sim[:,:,3:6]
    X_sim_residual[:,1:,9:12] = np.diff(X_sim_residual[:,:,6:9], axis=1)
    X_sim_residual[:,2:,12:15] = np.diff(X_sim_residual[:,1:,9:12], axis=1) 
    

    X_real_residual[:,:,:6] = X_real
    X_real_residual[:,:,6:9] = X_real[:,:,:3] - X_real[:,:,3:6]
    X_real_residual[:,1:,9:12] = np.diff(X_real_residual[:,:,6:9], axis=1)
    X_real_residual[:,2:,12:15] = np.diff(X_real_residual[:,1:,9:12], axis=1)
    

    return X_sim_residual, X_real_residual


X_sim_residual, X_real_residual = treat_data(X_sim,X_real)


def split_data(X,Y,test_size):
    # split the data into test and train
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = test_size,random_state=42,stratify=Y)

    return X_train, X_test, y_train, y_test

test_size_sim = 0.1
test_size_real = 0.2
X_sim_train, X_sim_test, Y_sim_train, Y_sim_test = split_data(X_sim_residual,Y_sim,test_size_sim)
X_real_train, X_real_test, Y_real_train, Y_real_test = split_data(X_real_residual,Y_real,test_size_real)

# Normalization
mean, std = normalize(X_sim_train)
X_sim_train_normalize  = (X_sim_train - mean)/std
X_sim_test_normalize = (X_sim_test - mean)/std


# We use datatensor 
X_sim_train_data = torch.tensor(X_sim_train_normalize, dtype=torch.float32)
Y_sim_train_data = torch.tensor(Y_sim_train, dtype=torch.long)
X_sim_test_data = torch.tensor(X_sim_test_normalize, dtype=torch.float32)
Y_sim_test_data = torch.tensor(Y_sim_test, dtype=torch.long)


# Dataloader
batch = 32
seq_len = 1000
features = 9

train_dataset = TensorDataset(X_sim_train_data, Y_sim_train_data)
val_dataset = TensorDataset(X_sim_test_data, Y_sim_test_data)

training_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True,num_workers=num_cpus)
validation_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False,num_workers=num_cpus)


#We use a CNN to do the training
model = ComplexCNN(input_size = 15, num_classes = 9, num_channels =32, kernel_size=2, dropout=0.2)

# We use the fonction loss CrossEntropy 
loss_fn = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

def train_one_epoch(epoch_index, tb_writer,data_loader):
    # we train the model on one epoch
    running_loss = 0.
    last_loss = 0.


    for i, data in enumerate(data_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            #print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(data_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss



# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 70

best_vloss = 1000000

# first training
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer,training_loader)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
    

# We evaluate the model with the precision and the confusion matrix
model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
  outpouts = model(X_sim_test_data)
  _, prediction = torch.max(outpouts,1)

y_prediction = prediction.cpu().numpy()
y_true = Y_sim_test_data.cpu().numpy()

fig, ax = plt.subplots(figsize=(10, 10))
ConfusionMatrixDisplay.from_predictions(y_true,y_prediction,display_labels=categories,xticks_rotation=45,ax=ax)
acc_sim = (y_prediction == y_true).mean()
plt.title(f"Confusion Matrix (Acc: {acc_sim:.2f})")

# On sauvegarde le fichier au lieu de l'afficher
plt.savefig('confusion_matrix_rf.png', bbox_inches='tight')



# Transfer Learning (Real)

# Dataloader
batch = 9
seq_len = 1000
features = 9

mean_real, std_real = normalize(X_real_train)
X_real_train_normalize  = (X_real_train - mean_real)/std_real
X_real_test_normalize = (X_real_test -mean_real)/std_real

X_real_train_data = torch.tensor(X_real_train_normalize, dtype=torch.float32)
Y_real_train_data = torch.tensor(Y_real_train, dtype=torch.long)
X_real_test_data = torch.tensor(X_real_test_normalize, dtype=torch.float32)
Y_real_test_data = torch.tensor(Y_real_test, dtype=torch.long)

train_real_dataset = TensorDataset(X_real_train_data, Y_real_train_data)
val_real_dataset = TensorDataset(X_real_test_data, Y_real_test_data)

training_real_loader = DataLoader(train_real_dataset, batch_size=batch, shuffle=True,num_workers=num_cpus)
validation_real_loader = DataLoader(val_real_dataset, batch_size=batch, shuffle=False,num_workers=num_cpus)
'''
for param in model.lstm.parameters():
    param.requires_grad = False  # gele le LSTM

for param in model.tcn.parameters():
    param.requires_grad = False
    
for param in model.fc.parameters():
    param.requires_grad = True
'''

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
best_vloss = 1000000
EPOCHS = 40
model_real_path = None

# second training
for epoch in range(EPOCHS):
    #print(f'TRANSFER EPOCH {epoch + 1}:')
    
    model.train(True)
    avg_loss = train_one_epoch(epoch, writer, training_real_loader)
    
    running_vloss = 0.0
    model.eval()
    
    with torch.no_grad():
        for i, vdata in enumerate(validation_real_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
    
    print(f'LOSS real train {avg_loss} valid {avg_vloss}')
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_real_path = 'model_real_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(),model_real_path)
        


# We evaluate the model with the precision and the confusion matrix
if model_real_path != None:
  model.load_state_dict(torch.load(model_real_path))
else : 
  model.load_state_dict(torch.load(model_path))

model.eval()

with torch.no_grad():
  outpouts = model(X_real_test_data)
  _, prediction = torch.max(outpouts,1)

y_prediction = prediction.cpu().numpy()
y_true = Y_real_test_data.cpu().numpy()

fig, ax = plt.subplots(figsize=(10, 10))
ConfusionMatrixDisplay.from_predictions(y_true,y_prediction,display_labels=categories,xticks_rotation=45,ax=ax)
acc_sim = (y_prediction == y_true).mean()
plt.title(f"Confusion Matrix (Acc: {acc_sim:.2f})")

# On sauvegarde le fichier au lieu de l'afficher
plt.savefig('confusion_matrix_real_rf.png', bbox_inches='tight')

#Faire de la cross validation et moyenne et ecart type pour la soutenace final
