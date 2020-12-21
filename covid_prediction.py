import csv
#from tensorflow import keras
import glob, os
import plaidml.keras
plaidml.keras.install_backend()
#Bibioteka do obsługi sieci neuronowych
import keras
#Załadowania bazy uczącej
import os
import numpy as np
from keras.models import load_model

Count= 62
Size= 7
BazaVec = np.empty((Count,Size,1))
BazaAns = np.empty((Count,1))
table={}
i=0
start_pl_index=12541
end_pl_index=12609
with open('data.csv', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        table[i]=row
        i=i+1
z1=0
for l in range(start_pl_index,end_pl_index-6):
    for j in range(7):
        BazaVec[z1,j,:]=float(table[l+j][4]) # dzienny przyrost 
    BazaAns[z1]=float(table[l+7][4])
    z1=z1+1
max_answer=np.amax(BazaAns)/2
print(max_answer)
BazaAns=BazaAns/max_answer-1
BazaAns=BazaAns[0:z1]
maxval = np.amax(BazaVec)/2
BazaVec=BazaVec/max_answer-1
BazaVec=BazaVec[0:z1,:,:]
#
##Stworzenia modelu sieci
inputt = keras.engine.input_layer.Input(shape=(Size,1),name="wejscie")
#
FlattenLayer = keras.layers.Flatten()
#
path = FlattenLayer(inputt)

for i in range(0,5):
  LayerDense1 = keras.layers.Dense(10, activation=None, use_bias=True, kernel_initializer='glorot_uniform')
  path = LayerDense1(path)
  LayerPReLU1 = keras.layers.PReLU(alpha_initializer='zeros', shared_axes=None)
  path = LayerPReLU1(path)
#
LayerDenseN = keras.layers.Dense(1,activation=None, use_bias=True, kernel_initializer='glorot_uniform')
output = LayerDenseN(path)
##---------------------------------
## Creation of TensorFlow Model
##---------------------------------
covidModel = keras.Model(inputt, output, name='covidEstimatior')
#
covidModel.summary() # Display summary
#
##Włączenia procesu uczenia
#
rmsOptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#
covidModel.compile(optimizer=rmsOptimizer,loss=keras.losses.mean_absolute_error)
#covidModel.compile(optimizer=rmsOptimizer,loss=keras.losses.binary_crossentropy,metrics=['accuracy'])
#

covidModel.fit(BazaVec, BazaAns, epochs=125, batch_size=10, shuffle=True)
covidModel.save('covid.h5')
##Przetestować / użyć sieci

BazaVecW = BazaVec[z1-1:z1,:,:]
covid = covidModel.predict(BazaVecW)
print((covid[0]+1)*max_answer)
val=(covid[0]+1)*max_answer
if os.path.exists('predict_next_value.txt'):
    os.remove('predict_next_value.txt')
filee=open('predict_next_value.txt','a+')
filee.write('Przewidywana liczba zachorowań na 11.05.2020: '+str(int(val)))
filee.close()