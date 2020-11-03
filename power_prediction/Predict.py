#!/usr/bin/python3
# Program to Predict the Weather Conditions -> Power network
# Refer to Train.py for model explanation
# You are probably looking for makePrediction(data, count, reset, powOnly)
from tensorflow.keras.models import Sequential, model_from_json
import datetime
import numpy as np
import json

# Scalars for data
GHISCALAR = 1875.0
SPDSCALAR = 21.0
TMPSCALAR = 58.0
POWSCALAR = 6470.0


# Train Control Variables
# -----------------------
FEATURES    = 6 # Input variables in a timestep
NUM_STEPS   = 15 # How many timesteps to consider at once

# Remember to save favorite model
# TODO pass model as filename?
model_file  = 'models/Power_Pred_model_2020_11_03_01_03.json'
weight_file = 'weights/Power_Pred_weights_2020_11_03_01_03.h5'

# LOAD FROM DISK
json_file = open(model_file, 'r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)
model.load_weights(weight_file)
print("Loaded model from disk")

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


# Use the scalars given from findmaxvalues.py
def downScale(data):
    # Apply scalars to make data NN readable
    # When this is called, the time are the last 4 values, integrate hr/min and month/day
    for t in range(NUM_STEPS):
        data[t][-4] = (data[t][-4] - 1) + ((data[t][-3] - 1)/31.0) # Month.(Day/31)
        #data[0][t][-4] = (data[0][t][-4] - 1) + ((data[0][t][-3] - 1)/31.0) # Month.(Day/31)
        data[t][-3] = (data[t][-2] - 1) + ((data[t][-1] - 1)/60.0) # Hour.(Minute/60)
        #data[0][t][-3] = (data[0][t][-2] - 1) + ((data[0][t][-1] - 1)/60.0) # Hour.(Minute/60)
    data = data[:,:-2] # Trim off 2 items for days and minutes being removed
    #data = data[0,:,:-2] # Trim off 2 items for days and minutes being removed
    output = np.c_[data[:,0]/GHISCALAR, # GHI
                   data[:,1]/360.0,     # WindDir
                   data[:,2]/SPDSCALAR, # WindSpd
                   data[:,3]/TMPSCALAR, # AmbTemp
                   data[:,4]/12.0,      # Month
                   data[:,5]/24.0]      # Hour
    output = output.reshape(1,NUM_STEPS,FEATURES)
    return output

def incrementTime(date):
    # Manually increment the data by 1 minute, from downscaled mode
    # Remember downscaled time is [(Month.day/31)/12, (Hour.minute/60)/24]
    year = int(datetime.datetime.now().strftime("%Y"))
    month = int(date[0] * 12) + 1
    day = int((round(date[0] * 12 % 1 * 31) + 1) % 24)
    hour = int(date[1]*24) + 1
    minute = int((round(date[1] * 24 % 1 * 60) + 1) % 60)
    if minute == 0 : hour = hour + 1 # O'clock increments are being funky >:(
    d = datetime.datetime(year, month, day, hour, minute, 0)
    print(d)
    d += datetime.timedelta(minutes=1)
    output = [(d.month + (d.day - 1)/31.0 - 1)/12.0, (d.hour + (d.minute - 1)/60.0 - 1)/24.0]
    return output

    # Simplified method: just increment hours by 1 minute
    # Saved in case of latency, just replace with this
    '''
    data[1] = data[1] + (1/60.0)      # Increment hour by 1 minute
    if data[1] >= 1:                  # if hours > 24,
        data[1] = 0                   # Roll over (Midnight)
    return data
    '''

def upScale(data):
    # Convert NN readable data back to proper units, separate months/days and hours/minutes
    output = np.c_[data[0,:,0]*GHISCALAR, # GHI
                   data[0,:,1]*360.0,     # WindDir
                   data[0,:,2]*SPDSCALAR, # WindSpd
                   data[0,:,3]*TMPSCALAR, # AmbTemp
    # Decouple Days from months and minutes from hours
                   [int(x * 12) + 1 for x in data[0,:,4]],              # Month
                   [round(x * 12 % 1 * 31) + 1 for x in data[0,:,4]],   # Day
                   [int(x * 24) + 1 for x in data[0,:,5]],              # Hour
                   [round(x * 24 % 1 * 60) + 1 for x in data[0,:,5]],   # Minute
                   data[0,:,6]*POWSCALAR] # Power
    output = output.reshape(1,NUM_STEPS,FEATURES + 3)
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

def displayPrediction(data):
    # TODO merge with upper display function
    for y in data:
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
def makePrediction(data, count, scale=True, reset=True, powOnly=True):
    # Outputs predictions of future data and power, iterated to #count
    # powOnly chooses outputs weather predictions (False) or just power (True)
    #NOTE requires time series formatted data in same dimensions
    # as model was trained

    #TODO TODO Fix data before feeding back in for stability

    if reset:
        model.reset_states()

    if scale:
        nextdata = downScale(data)
    power = []
    fulloutput = []
    for i in range(count):
        # Predict and build the output options
        new_pred = model.predict(nextdata)

        #Make sure the date is consecutive, save last time and increment seperately
        last_time = nextdata[0,:,-2:]
        for t in range(NUM_STEPS):
            next_time = incrementTime(last_time[t])
            new_pred[0,t,-3] = next_time[0] # Indices look odd because power is appended
            new_pred[0,t,-2] = next_time[1]

        fulloutput.append(upScale(new_pred)[0][-1])
        power.append(new_pred[0][-1][-1] * POWSCALAR)

        # Create the dataset for the next loop
        # Roll data left (pops off oldest set)
        nextdata = np.roll(nextdata,-1,axis=1)
        # Append newest prediction, without power for input data
        nextdata[0][-1] = new_pred[0][-1][0:-1]
    if powOnly:
        return power
    else:
        return fulloutput


# Testing with some data from 2018
# April 11, 15:51-15:53
'''
testinput1 = np.array(
        [[[855.19,48.00,15*.44704,(75.75-32)/1.8,4,11,15,51],
          [852.36,38.33,14*.44704,(75.73-32)/1.8,4,11,15,52],
          [849.53,28.66,13*.44704,(75.71-32)/1.8,4,11,15,53],
          [864.66,34.20,13*.44704,(75.63-32)/1.8,4,11,15,54],
          [843.66,39.74,14*.44704,(75.55-32)/1.8,4,11,15,55]]])
# April 11, 15:54-16:09
testinput2 = np.array(
        [[846.66,34.19,13*.44704,(75.63-32)/1.8,4,11,15,54],
          [843.79,39.74,13*.44704,(75.55-32)/1.8,4,11,15,55],
          [840.92,45.27,13*.44704,(75.55-32)/1.8,4,11,15,56],
          [838.05,50.81,13*.44704,(75.55-32)/1.8,4,11,15,57],
          [837.39,42.90,13*.44704,(75.55-32)/1.8,4,11,15,58],
          [836.72,34.99,13*.44704,(75.55-32)/1.8,4,11,15,59],
          [834.55,27.08,13*.44704,(75.55-32)/1.8,4,11,16,0],
          [833.06,26.04,13*.44704,(75.55-32)/1.8,4,11,16,1],
          [831.56,39.70,13*.44704,(75.55-32)/1.8,4,11,16,2],
          [830.07,39.70,13*.44704,(75.55-32)/1.8,4,11,16,3],
          [825.41,39.70,13*.44704,(75.55-32)/1.8,4,11,16,4],
          [820.75,39.70,13*.44704,(75.55-32)/1.8,4,11,16,5],
          [816.09,39.70,13*.44704,(75.55-32)/1.8,4,11,16,6],
          [813.72,50.81,14*.44704,(75.40-32)/1.8,4,11,16,7],
          [811.35,50.81,14*.44704,(75.40-32)/1.8,4,11,16,8]])
#Testing the function
test = makePrediction(testinput2, 3, reset=True, powOnly=False)
print("PREDICTED DATA")
displayPrediction(test)
'''
