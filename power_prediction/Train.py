#!/usr/bin/python3
# Program to Train the Weather Conditions -> Power network
# Uses LSTM with time series formatting
    # ex: [0,1,2,3,4] -> [[0,1,2],[1,2,3],[2,3,4]]
# Each time slice predicts its future state + power at that time
    #
    #   <data1> <data2> <*data3*> <= Predicts 1 beyond its input
    #     ^       ^       ^             -Loop using this data for
    #   [LSTM]->[LSTM]->[LSTM]           further predictions
    #     ^       ^       ^
    #   <data0> <data1> <data2>
    #
# For further help Check 
# https://stackoverflow.com/questions/38714959/understanding-keras-lstms
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas# ONLY USED FOR CSV<REPLACE
import datetime
# Following lines needed for gpu training
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

date = datetime.datetime.now()
formatdate = date.strftime('%Y_%m_%d_%H_%M')

'''
Train Control Variables
'''
runtitle    = "Combined time groups, 50 epochs full after 100 small, 15 steps"
FEATURES    = 6 # Input variables in a timestep

NUM_STEPS   = 15 # How many timesteps to consider at once
NUM_BATCHES = 240
NUM_EPOCHS  = 50
# NOTE Always check if you're training with proper data
fulldataset = True
load_model = True

if fulldataset:
    train_file  = 'olddata/2019_Interpolated_Data.csv'# All 2019 (SLOW)
    test_file  = 'olddata/2018_Interpolated_Data.csv' # All 2018 (SLOW)
else:
    train_file = 'olddata/Jan_interpolated.csv' # Just Jan for fast proto
    test_file = 'olddata/Feb_interpolated.csv'  # Just Feb for fast proto
''' File Control '''
# Choose to load a model to continue training
load_model_file = 'models/Power_Pred_model_2020_11_02_23_31.json'
load_weight_file = 'weights/Power_Pred_weights_2020_11_02_23_31.h5'
load_history_file = 'history/Power_Pred_history_2020_11_02_23_31.csv'
model_file  = 'models/Power_Pred_model_' + formatdate + '.json'
weight_file = 'weights/Power_Pred_weights_' + formatdate + '.h5'
history_file = 'history/Power_Pred_his_' + formatdate + '.csv'

'''
Form the training data and testing data #
 Load from csv, Sum total power output,
 convert Temperature to Celsius and Wind Speed to m/s
 Get Time of year/day data 
'''
# NOTE Power data headers altered to remove special characters.
def file_to_data(csv):
    #TODO uses pandas to open a csv, when next changing data, use csv api
    loadcsv = pandas.read_csv(csv)
    # Sum production meter output
    power_output = loadcsv['LF_Prod1_kWac'].add(loadcsv['LF_Prod2_kWac']) 
    power_output = power_output.add(loadcsv['LF_Prod3_kWac'])
    power_output = power_output.add(loadcsv['LF_Prod4_kWac'])
    GHI_data = loadcsv['LF_WS2_GHI W/m']
    windDir_data = loadcsv['LF_WS1_WindDir']
    def MPHtoMPS(mph):
        return mph * 0.44704
    windSpd_data = loadcsv['LF_WS1_WindSpd mph'].apply(MPHtoMPS)
    def toCelsius(Fahrenheit):
        return (Fahrenheit - 32) / 1.8
    ambTemp_data = loadcsv['LF_WS1_TempAmb F'].apply(toCelsius)
    month = [] ; hour = [] #; minute = [] ; day = []
    for i in range(len(loadcsv)): # Combine hours + minutes, months + days, index at 0 so subtract 1
        month.append((float(str(loadcsv['Timestamp'][i])[5:7]) - 1 + 
            (float(str(loadcsv['Timestamp'][i])[8:10]) - 1)/31.0)/12.0)
        #day.append(float(str(loadcsv['Timestamp'][i])[8:10])/31.0)
        hour.append((float(str(loadcsv['Timestamp'][i])[11:13]) - 1 +
            (float(str(loadcsv['Timestamp'][i])[14:16]) - 1)/60.0)/24.0)


    # NOTE Scaled by values found in findmaxvalues.py for data
    # Scalars are maximums of data file + 30% for safety
        # Sub Note: That margin of safety should be checked for
        # its effects on accuracy TODO easy test just divide 1.3 mult 1.1 right here
    GHI_SCALAR = 1875.0   # W/m
    WINDDIR_SCALAR = 360.0# deg
    WINDSPD_SCALAR = 21.0 # mps (?)
    AMBTEMP_SCALAR = 58.0 # C
    POWER_SCALAR = 6470.0 # kWac

    scaled_GHI = GHI_data / GHI_SCALAR
    scaled_windDir = windDir_data / WINDDIR_SCALAR
    scaled_windSpd = windSpd_data / WINDSPD_SCALAR
    scaled_temp = ambTemp_data / AMBTEMP_SCALAR
    scaled_power = power_output / POWER_SCALAR


    # NOTE NOTE Here is where the data is organized into how it
    # will train and predict. New data sources are added here
    # (After being scaled to between 0 and 1) in the above 

    # NOTE the power data is not available continuously,
    # can not have power as an input parameter.
    # np.c_ appends columns, change this to add/remove features
    input_data = np.c_[scaled_GHI,scaled_windDir,scaled_windSpd,
            scaled_temp,month,hour]
            #scaled_temp,month,day,hour,minute]
    output_data = np.c_[scaled_GHI,scaled_windDir,scaled_windSpd,
            scaled_temp,month,hour,scaled_power]
            #scaled_temp,month,day,hour,minute,scaled_power]
    return input_data, output_data


    # How to format data for the network
def toTimeSeries(inputData, timesteps, batches=-1, start=0):
    ''' Data must be formatted in time series:
    data = [0,1,2,3,4,5] becomes [[0,1,2,3], [1,2,3,4], [2,3,4,5]]
    timesteps determines the size of the window, in the example it's 4
    Use start to offset beginning, and batches to choose how many 
    minutes to use beyond start
    '''
    # Trim if reaches past given data's bounds,
    if ((batches + start > len(inputData) - timesteps - start) or
            (batches < 0)):
         batches = len(inputData) - timesteps - start
    # Build Output
    timeSeries = []
    for t in range(start, start + batches ):
        newRow = []
        for dt in range(timesteps):
            newRow.append(inputData[t + dt])
        timeSeries.append(newRow)
    # Output has shape (batches,timesteps,features)
    return np.array(timeSeries)

# create Training data
# The filenames are specified in header
train_input_data,train_output_data = file_to_data(train_file)
test_input_data,test_output_data = file_to_data(test_file)

# Remember to start y a step ahead to output t+1
# also stop x a step short to make final prediction novel
x_train = toTimeSeries(train_input_data[0:-1], timesteps=NUM_STEPS)
y_train = toTimeSeries(train_output_data[1:], timesteps=NUM_STEPS)
x_test = toTimeSeries(test_input_data[0:-1], timesteps=NUM_STEPS)
y_test = toTimeSeries(test_output_data[1:], timesteps=NUM_STEPS)


'''
Build the Model
'''
if load_model:
    # LOAD FROM DISK
    json_file = open(load_model_file, 'r')
    loaded_model = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model)
    model.load_weights(load_weight_file)
    print("Loaded model from disk")
else:
    inputs = Input(shape=(NUM_STEPS,FEATURES))
    lstm1 = LSTM(FEATURES, input_shape = (NUM_STEPS,FEATURES),
        return_sequences=True)(inputs)
    #drops = Dropout(.2)(lstm1) # todo evaluate
    lstm2 = LSTM(36, input_shape = (NUM_STEPS,FEATURES),
        return_sequences=True)(lstm1) # (drops)
    #lstm3 = LSTM(36, input_shape = (NUM_STEPS,FEATURES),
        #return_sequences=True)(lstm2) 
#    lstm4 = LSTM(12, input_shape = (NUM_STEPS,FEATURES),
        #return_sequences=True)(lstm3) 
    outputs = Dense(FEATURES+1, activation='tanh')(lstm2)
    model = Model(inputs=inputs, outputs=outputs)
    print("New model instance")

#TODO check this out
#model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=["accuracy"]) # Evaluate rmsprop
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

#Dynamic learning rate scheduler
#From https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def step_decay(epoch):
    initial_lrate = 0.1
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# Train the model
# Done as loop to ensure states don't bleed through end of epochs
# Keeps internal "memory" chronological
loss_train = []
loss_val = []
graphepochs = range(1,NUM_EPOCHS+1)
for e in range(NUM_EPOCHS):
    print("Epoch ", e + 1, " / ", NUM_EPOCHS)
    # TODO set learning rate
    # Look into "dynamic learning rate"
    history = model.fit(x_train,y_train, epochs=1,
            batch_size=NUM_BATCHES, shuffle=False,
            validation_data=(x_test,y_test))
    #print("KEYS:")
    #print(history.history.keys())
    loss_train.append(history.history['loss'])
    loss_val.append(history.history['val_loss'])
    model.reset_states()

# Make graphs of training info
plt.figure(0)
plt.plot(graphepochs, loss_train, 'b', label='Training Loss')
plt.plot(graphepochs, loss_val, 'r', label='Validation Loss')
plt.title(runtitle)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaulate the model
score = model.evaluate(x_test,y_test, verbose = 0)
print('Test Loss:', score)

# Save model to files in header
model_json = model.to_json()
with open(model_file, 'w') as json_file:
    json_file.write(model_json)
model.save_weights(weight_file)
# Write down the name of the best model so far for backup
print("Saved model to disk")
print("Model: " + model_file)
print("Weight: " + weight_file)
print("History: " + history_file)

#Save the history for expanding or transfer learning models


# For programs using this model:
''' LOAD FROM DISK
json_file = open(model_file, 'r')
loaded_model = json_file_read()
json_file.close()
model = model_from_json(loaded_model)
model.load_weights(weight_file)
print("Loaded model from disk")
'''
        
