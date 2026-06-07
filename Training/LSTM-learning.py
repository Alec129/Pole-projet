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

link = '../mydataset/my_dataset_train.mat'
link_test = '../mydataset/my_dataset_test.mat'

X_sim, Y_sim, X_real, Y_real = import_data(link,link_test)



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

    return (data - mean)/std

def treat_data(X_sim,X_real):
    # add the residual to vector (residual = trajectory without error - trajectory with error)
    dim_sim = np.shape(X_sim)
    dim_real = np.shape(X_real)

    X_sim_residual = np.zeros((dim_sim[0],dim_sim[1],dim_sim[2] + 3))
    X_real_residual = np.zeros((dim_real[0],dim_real[1],dim_real[2] + 3))

    X_sim_residual[:,:,:6] = X_sim
    X_sim_residual[:,:,6:9] = X_sim[:,:,:3] - X_sim[:,:,3:6]

    X_real_residual[:,:,:6] = X_real
    X_real_residual[:,:,6:9] = X_real[:,:,:3] - X_real[:,:,3:6]

    return X_sim_residual, X_real_residual


X_sim_residual, X_real_residual = treat_data(X_sim,X_real)

'''
print("tensor X:",np.shape(X_sim_residual))
print("tensor Y:", Y_sim)
print("tensor X real:", np.shape(X_real_residual))
print("tensor Y real:", np.shape(Y_real))
'''

def split_data(X,Y,test_size):
    # split the data into 
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = test_size,random_state=42,stratify=Y)

    return X_train, X_test, y_train, y_test

test_size_sim = 0.1
test_size_real = 0.2
X_sim_train, X_sim_test, Y_sim_train, Y_sim_test = split_data(X_sim_residual,Y_sim,test_size_sim)
X_real_train, X_real_test, Y_real_train, Y_real_test = split_data(X_real_residual,Y_real,test_size_real)

'''
print(np.shape(X_sim_train))
print(np.shape(Y_sim_test))
print(np.unique(Y_sim_train, return_counts=True))
print(np.unique(Y_sim_test, return_counts=True))
'''

X_sim_train_normalize  = normalize(X_sim_train)
X_sim_test_normalize = normalize(X_sim_test)
X_real_train_normalize  = normalize(X_real_train)

'''
print(np.shape(X_sim_train_normalize))
print(np.shape(X_real_train_normalize))
'''



# We use datatensor 
X_sim_train_data = torch.tensor(X_sim_train_normalize, dtype=torch.float32)
Y_sim_train_data = torch.tensor(Y_sim_train, dtype=torch.long)
X_sim_test_data = torch.tensor(X_sim_test_normalize, dtype=torch.float32)
Y_sim_test_data = torch.tensor(Y_sim_test, dtype=torch.long)

'''
X_sim_train_flat = X_sim_train.reshape(X_sim_train.shape[0], -1)
Y_sim_flat = Y_sim_train.ravel()
X_sim_test_flat  = X_sim_test.reshape(X_sim_test.shape[0], -1)
Y_sim_test_flat = Y_sim_test.ravel()

X_real_train_flat = X_real_residual.reshape(X_real_residual.shape[0], -1)
Y_real_flat = Y_real.ravel()
'''


# Dataloader
batch = 32
seq_len = 1000
features = 9

train_dataset = TensorDataset(X_sim_train_data, Y_sim_train_data)
val_dataset = TensorDataset(X_sim_test_data, Y_sim_test_data)

training_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True,num_workers=num_cpus)
validation_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False,num_workers=num_cpus)


#We use a LSTM to do the training
model = LSTMModel(
    input_size=9,        # features
    hidden_size=64,
    num_classes = 9
)

# We use the fonction loss CrossEntropy 
loss_fn = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch(epoch_index, tb_writer):
    # we train the model on one epoch
    running_loss = 0.
    last_loss = 0.


    for i, data in enumerate(training_loader):
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
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss



# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 50

best_vloss = 1000000

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


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
