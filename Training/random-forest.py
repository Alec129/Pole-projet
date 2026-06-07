import numpy as np
import matplotlib.pyplot as plt
import scipy 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


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


def import_data(link,link_test):
    # import the data from the dataset and return the vectors of the data and the label
    mat = scipy.io.loadmat(link)
    mat_test = scipy.io.loadmat(link_test)

    X_sim = np.array(mat['X_array'][0])
    data_X = np.array([X_sim[i] for i in range(len(X_sim))])
    Y_sim = np.array(mat['y_array'][0])
    data_Y = np.array([Y_sim[i] for i in range(len(Y_sim))])

    X_real = np.array(mat_test['X_test_array'][0])
    data_test_X = np.array([X_real[i] for i in range(len(X_real))])
    Y_real = np.array(mat_test['y_test_array'][0])
    data_test_Y = np.array([Y_real[i] for i in range(len(Y_real))])

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

print(X_sim)
print("tensor X:", np.shape(X_sim))
print("tensor Y:", np.shape(Y_sim))
print("tensor X real:", np.shape(X_real))
print("tensor Y real:", np.shape(Y_real))

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

print("tensor X:",np.shape(X_sim_residual))
print("tensor Y:", np.shape(Y_sim))
print("tensor X real:", np.shape(X_real_residual))
print("tensor Y real:", np.shape(Y_real))


def split_data(X,Y,test_size):
    # split the data into 
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = test_size,random_state=42,stratify=Y)

    return X_train, X_test, y_train, y_test

test_size_sim = 0.1
test_size_real = 0.2
X_sim_train, X_sim_test, Y_sim_train, Y_sim_test = split_data(X_sim_residual,Y_sim,test_size_sim)
X_real_train, X_real_test, Y_real_train, Y_real_test = split_data(X_real_residual,Y_real,test_size_real)

print(np.shape(X_sim_train))
print(np.shape(Y_sim_test))
print(np.unique(Y_sim_train, return_counts=True))
print(np.unique(Y_sim_test, return_counts=True))

X_sim_train_normalize  = normalize(X_sim_train)
X_real_train_normalize  = normalize(X_real_train)
print(np.shape(X_sim_train_normalize))
print(np.shape(X_real_train_normalize))



X_sim_train_flat = X_sim_train.reshape(X_sim_train.shape[0], -1)
Y_sim_flat = Y_sim_train.ravel()
X_sim_test_flat  = X_sim_test.reshape(X_sim_test.shape[0], -1)
Y_sim_test_flat = Y_sim_test.ravel()

X_real_train_flat = X_real_residual.reshape(X_real_residual.shape[0], -1)
Y_real_flat = Y_real.ravel()


#train the model on simulation data
rf = RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)
rf.fit(X_sim_train_flat, Y_sim_train)

#test the model on simulation and real data
y_pred = rf.predict(X_sim_test_flat)
acc_sim = rf.score(X_sim_test_flat, Y_sim_test_flat)
print("Accuracy sim:", acc_sim)
ConfusionMatrixDisplay.from_predictions(
    Y_sim_test_flat,
    y_pred,
    display_labels=categories,
    xticks_rotation=45
)
plt.show()


y_pred_real = rf.predict(X_real_train_flat)
acc_sim = rf.score(X_real_train_flat, Y_real_flat)
print("Accuracy test:", acc_sim)
ConfusionMatrixDisplay.from_predictions(
    Y_real_flat,
    y_pred_real,
    display_labels=categories,
    xticks_rotation=45
)

plt.show()
