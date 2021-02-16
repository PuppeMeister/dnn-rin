from keras.layers import Dense, Bidirectional, LSTM, TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras.initializers import RandomNormal
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix as cm
#fix the value of the random seed, so that the code runs same each time
np.random.seed(5)
#define the length of data based on which the prediction is to happen
t = 3
#read the dataset in the form of 9 time steps x 5 features
dataset = pd.read_csv('Group 7 - A.csv')
X = dataset.iloc[:, 2:7].values
scaler = StandardScaler().fit(X)
#define X_train as first 3 time steps and Y_train as the consecutive 1 time step, so the sequences shifts by one every time
#resulting in (9-t)=6 shifts x 10 people samples = 60 samples
X_train = np.ndarray((60,3,5),np.float32)
Y_train = np.ndarray((60,1,5),np.float32)
for i in range(0,10):
    for j in range(0,9-t):
        X_train[int(i*(9-t)+j)] = X[(i*9+j):(i*9+j+3),:]
        Y_train[int(i*(9-t)+j)] = X[(i*9+j+3),:]
Y_train.resize((60,5))
#the initial dataset is reshaped into 10 people x 9 time steps x 5 features
#so that predicted synthetic data can be added to each person
data = np.ndarray((10,9,5),np.float32)
for i in range (0,10):
    data[i] = X[i*9:(i+1)*9, :]
#DNN model for classification
def model(shape):
    model = Sequential()
    model.add(Dense(60, input_dim=shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model
#LSTM model for prediction
pred_optim = SGD(momentum=0.9)
def pred_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(4, activation='relu', input_shape=(None, 5), return_sequences=True, kernel_initializer=RandomNormal(stddev=0.075))))
    model.add(Bidirectional(LSTM(3, activation='relu')))
    model.add(Dense(5))
    model.compile(optimizer='adam', loss='mse')
    return model
#This function predicts new data points, where n_loop is the number of new data points to be predicted
#and the output is the augmented data set in the shape 10 x (9 + n_loop) x 5

callback = EarlyStopping(monitor = 'val_loss', patience = 30, restore_best_weights=True)
def pred_loop(X,Y,n_loop,data):
    #Print All Parameter
    print("Going to Train Augmenting Model ")
    print("n_loop (desirable number of synthetic data) = ", n_loop)
    print("Type of data = ", type(data))
    print("Type of X = ", type(X))
    print("Type of Y = ", type(Y))

    print("This is X ----")
    print(X)
    print("________________________")

    print("This is Y ----")
    print(Y)

    print("_______________________")
    pred = pred_model()
    pred.fit(X, Y, epochs=1000, validation_split=0.2, batch_size=3, callbacks=callback)
    pred.summary()

    print("This is Data")
    print(data)
    print("_______________________________")
    for i in range(n_loop):
        print("Insert the predicting loop. Round ", i)
        print("data[:,",(6+i),":",data.shape[1],":]")
        print(data[:,(6+i):data.shape[1],:])

        predicted = pred.predict(data[:,(6+i):data.shape[1],:])
        predicted = predicted.reshape((10,1,5))
        #print(data.shape)
        data = np.concatenate((data,predicted),axis=1)

    print(data.shape)

    return data
        
augmented = pred_loop(X_train,Y_train,12,data)
"""
print(" ---------------- Augmentation Result[0] ---------------- ")
print(augmented[0])
print(" ---------------- Augmentation Result[1] ---------------- ")
print(augmented[1])
print(" ---------------- Augmentation Result[2] ---------------- ")
print(augmented[2])
print(" ---------------- Augmentation Result[3] ---------------- ")
print(augmented[3])
print(" ---------------- Augmentation Result[4] ---------------- ")
print(augmented[4])
print(" ---------------- Augmentation Result[5] ---------------- ")
print(augmented[5])
print(" ---------------- Augmentation Result[6] ---------------- ")
print(augmented[6])
print(" ---------------- Augmentation Result[7] ---------------- ")
print(augmented[7])
print(" ---------------- Augmentation Result[8] ---------------- ")
print(augmented[8])
print(" ---------------- Augmentation Result[9] ---------------- ")
print(augmented[9])
"""
in_size = augmented.shape

"""
print(" ---------------- Shape of Augmented Data before Reshape ---------------- ")
print(in_size)
#reshape the data back to (10 x no. of time steps) x 5 features
print(" ---------------- This value in_size[0]*in_size[1] (First index of reshape)  ---------------- ")
print(in_size[0]*in_size[1])
print(" ---------------- This value in_size[2] (Second index of reshape)  ---------------- ")
print(in_size[2])
"""

#Assign Augmented data to variable X
X = augmented.reshape(in_size[0]*in_size[1],in_size[2])

"""
print(" ---------------- The shape of Augmented Data Array after Reshape ---------------- ")
print(X.shape)
print(" ---------------- Type of X---------------- ")
print(type(X))
print(" ---------------- Augmented Data Array as X after Reshape  ---------------- ")
"""
for idx_x in range(210):
    print(X[idx_x])

#save the new augmented dataset in the initial format
#np.savetxt("new_data_rnn.csv",np.around(X,decimals=2),delimiter=",")
#scale the dataset into values between 0 to 1
X = scaler.transform(X)
#define first user as of class 1 and others as class 0
Y = np.concatenate((np.ones(in_size[1]),np.zeros(in_size[1]*(in_size[0]-1))))
print(" ---------------- Y shape = ", Y.shape)
#use 2/3 of the time steps for each person for training and 1/3 of the time steps for testing
X_train = X[0:int(in_size[1]*2/3), :]
y_train = Y[0:int(in_size[1]*2/3),None]
X_test = X[int(in_size[1]*2/3):int(in_size[1]), :]
y_test = Y[int(in_size[1]*2/3):int(in_size[1]),None]
for i in range(1,10):
    X_train = np.concatenate((X_train, X[i*in_size[1]:int(i*in_size[1]+in_size[1]*2/3), :]), axis = 0)
    X_test = np.concatenate((X_test, X[int(i*in_size[1]+in_size[1]*2/3):(i+1)*in_size[1], :]), axis = 0)
    y_train = np.concatenate((y_train , Y[i*in_size[1]:int(i*in_size[1]+in_size[1]*2/3),None]), axis = 0)
    y_test = np.concatenate((y_test, Y[int(i*in_size[1]+in_size[1]*2/3):(i+1)*in_size[1],None]), axis = 0)

#define the DNN model and train it
model=model(X_train.shape)
model.summary()
#optim = Adam(learning_rate = 0.0004)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=3, shuffle=True)

#Test the model
y_pred = model.predict(X_test)
#Hard labeling where labels of value less than 0.5 are classified as 0
threshold = 0.5
for i in range(len(y_pred)):
    if y_pred[i]<threshold:
        y_pred[i]=int(0)
    else:
        y_pred[i]=int(1)
print(y_pred)
#print the confusion matrix
print(cm(y_test,y_pred))


