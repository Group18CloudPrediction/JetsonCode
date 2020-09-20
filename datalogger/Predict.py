#!/usr/bin/python3
# Program to Predict the Weather Conditions -> Power network
# Refer to Train.py for model explanation
# You are probably looking for makePrediction(data, count, reset, powOnly)
from tensorflow.keras.models import Sequential, model_from_json
import numpy as np
import json

# Scalars for data
GHISCALAR = 1875.0
SPDSCALAR = 21.0
TMPSCALAR = 58.0
POWSCALAR = 6470.0


# Train Control Variables
# -----------------------
features    = 8 # Input variables in a timestep
num_steps   = 3 # How many timesteps to consider at once
# TODO choose final trained model for this,
# or make that an argument for the webapp to control

# Remember to save favorite model
model_file  = 'models/Power_Pred_model_2020_09_20_17_20.json'
weight_file = 'weights/Power_Pred_weights_2020_09_20_17_20.h5'

# LOAD FROM DISK
json_file = open(model_file, 'r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)
model.load_weights(weight_file)
print("Loaded model from disk")

# Testing with some data from 2018
# April 11, 15:51-15:53
#testinput1 = np.array(
#        [[[855.19,48.00,15*.44704,(75.75-32)/1.8,4,11,15,51],
#          [852.36,38.33,14*.44704,(75.73-32)/1.8,4,11,15,52],
#          [849.53,28.66,13*.44704,(75.71-32)/1.8,4,11,15,53]]])
# April 11, 15:54-15:56
#testinput2 = np.array(
#        [[[845.66,34.20,13*.44704,(75.63-32)/1.8,4,11,15,54],
#          [843.79,39.70,13*.44704,(75.55-32)/1.8,4,11,15,55],
#          [838.05,50.81,14*.44704,(75.40-32)/1.8,4,11,15,56]]])



# Use the scalars given from findmaxvalues.py
# NOTE being evaluated, subject to change
def downScale(data):
    # Apply scalars
    output = np.c_[data[0,:,0]/GHISCALAR, # GHI
                   data[0,:,1]/360.0,     # WindDir
                   data[0,:,2]/SPDSCALAR, # WindSpd
                   data[0,:,3]/TMPSCALAR, # AmbTemp
                   data[0,:,4]/12.0,      # Month
                   data[0,:,5]/31.0,      # Day
                   data[0,:,6]/24.0,      # Hour
                   data[0,:,7]/60.0]      # Minute
    output = output.reshape(1,3,8)
    return output

def upScale(data):
    output = np.c_[data[0,:,0]*GHISCALAR, # GHI
                   data[0,:,1]*360.0,     # WindDir
                   data[0,:,2]*SPDSCALAR, # WindSpd
                   data[0,:,3]*TMPSCALAR, # AmbTemp
                   data[0,:,4]*12.0,      # Month
                   data[0,:,5]*31.0,      # Day
                   data[0,:,6]*24.0,      # Hour
                   data[0,:,7]*60.0,      # Minute
                   data[0,:,8]*POWSCALAR] # Power
    output = output.reshape(1,3,9)
    return output

def display(data):
    # Quick and Dirty Display of output
    for x in data:
        for y in x:
            print("Date:    " , int(round(y[4])), int(round(y[5]))
                    , int(round(y[6])), int(round(y[7])))
            print("GHI:     " , y[0])
            print("WindDir: " , y[1])
            print("WindSpd: " , y[2])
            print("Temp:    " , y[3])
            if len(y) == 9:
                print("Power:   " , y[8])
            print()
                
 
# NOTE Important function
def makePrediction(data, count, reset=True, powOnly=True):
    # Outputs predictions of future data and power, iterated to #count
    # powOnly chooses outputs weather predictions (False) or just power (True)
    #NOTE requires time series formatted data in same dimensions
    # as model was trained
    if reset:
        model.reset_states()

    nextdata = downScale(data)
    power = []
    fulloutput = []
    for i in range(count):
        # Predict and build the output options
        new_pred = model.predict(nextdata)
        fulloutput.append(upScale(new_pred)[0][-1])
        power.append(new_pred[0][-1][-1] * POWSCALAR)

        # Create the dataset for the next loop
        # Roll data left (pops off oldest set)
        nextdata = np.roll(nextdata,-1,axis=2)
        # Append newest prediction, without power for input data
        nextdata[0][-1] = new_pred[0][-1][0:-1]
    if powOnly:
        return power
    else:
        print(np.array(fulloutput).shape)
        return fulloutput
#####################################3333
'''
def json_predict(jsonin):
    # Test wrapper, try integrating as argument to makePrediction
    featurelist = json.loads(jsonin)
    for item in featurelist:
        pass
    makePrediction(obj['data'])
    output = {}
    #output['predictedPower'] = # Grab list from array
    #output['predictedPower'] = # Grab list from array
    return output
'''

    
#####################################3333

#GHI, spd, dir, temp, mon, day, min
'''testinput1 = np.array(
        [[[855.19,48.00,15*.44704,(75.75-32)/1.8,4,11,15,51],
          [852.36,38.33,14*.44704,(75.73-32)/1.8,4,11,15,52],
          [849.53,28.66,13*.44704,(75.71-32)/1.8,4,11,15,53]]])
dicttest = {
    "GHI" : [855.19, 852.36,849.53],
    "windDir" : [48.00,38.33,28.66],
    "windSpd" : [21.458,17.135,12.812],
    "ambTemp" : [24.306,24.294,24.283]
    }
jsonout = json.dumps(dicttest)
jsondata = json.loads(jsonout)
#def jsonToInput(jsondata)
output = {}
timesteps = 3
for t in range(timesteps):
    for object in jsondata:
        #print(jsondata[object][t])
        output[t].append(jsondata[object][t])
        # How to add data in time series cleverly?
exit()'''


#####################################3333






# Just experimenting with the functions so far
'''
print("Input data")
display(testinput1)
print("Output data")
print("Full")
pred1 = makePrediction(testinput1, 2, True, False)
print(len(pred1))
'''
'''
print("Date:    " , int(round(pred1[4])), int(round(pred1[5]))
        , int(round(pred1[6])), int(round(pred1[7])))
print("GHI:     " , pred1[0])
print("WindDir: " , pred1[1])
print("WindSpd: " , pred1[2])
print("Temp:    " , pred1[3])
print("Power:   " , pred1[8])
print()
'''
'''
print(pred1)
print("Pow only")
pred1 = makePrediction(testinput1, 2, True, True)
print(pred1)
exit()

pred2 = makePrediction(testinput2, 1, False, False)
'''
'''
print(testinput1[0,1,:])
print(pred1[0,0,:])
print(testinput2[0,2,:])
print(pred2[0,1,:])
'''
#display(pred1)
#display(pred2)


