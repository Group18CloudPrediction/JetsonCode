#testing for cloud shadow cutoff value
import time
import os
from datalogger.datalogger import Datalogger

sleep_time = 10
datalogger = Datalogger('/dev/ttyS5') #path will need to change per system
total_ghi = 0
runs = 100
cur_run = 0

if os.path.exists("no_clouds_measurements.txt"):
  os.remove("no_clouds_measurements.txt")

while(cur_run < runs):
	print("run #" + str(cur_run))
	starttime = time.time()
	datalogger.poll() # get the current weather data to use for the next prediction
	ghi = datalogger.weather_data.slrFD_W
	f = open("no_clouds_measurements.txt", "a")
	f.write("run: " + str(cur_run) + " ... value: " + str(ghi) + "\n")
	f.close()
	total_ghi = total_ghi + ghi
	cur_run = cur_run + 1
	time.sleep(sleep_time - ((time.time() - starttime) % sleep_time))

average_ghi = total_ghi / runs
f = open("no_clouds_measurements.txt", "a")
f.write("average ghi: " + str(average_ghi))
f.close()
