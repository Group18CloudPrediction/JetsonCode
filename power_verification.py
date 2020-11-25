import minimalmodbus as mmbus
import serial
import time
import pymongo
import jsonpickle
import config.creds as creds
import config.substation_info as substation
import power_prediction.Predict as Predict
# import power_prediction.Train as Train

from datalogger.cloud_height_data import CloudHeightData
from datalogger.datalogger import Datalogger
from threading import Thread, Event
from datetime import datetime, timedelta

# Global access to the previous weather data
weather_data_list = []

# How many minutes of data to store in weather_data_list
# This is dependant on how the ML model was trained
mins_of_previous_data = 15

# Pull the relevant data needed for power prediction verification
def format_current_weather_data(datalogger):
	temp = []
	temp.append(datalogger.weather_data.slrFD_W)
	temp.append(datalogger.weather_data.windDir)
	temp.append(datalogger.weather_data.ws_ms)
	temp.append(datalogger.weather_data.airT_C)
	temp.append(datalogger.weather_data.poll_time.month)
	temp.append(datalogger.weather_data.poll_time.day)
	temp.append(datalogger.weather_data.poll_time.hour)
	temp.append(datalogger.weather_data.poll_time.minute)

	return temp

# Adds the latest minute's weather data to the front of the list, shifting
# everything else back, and dropping the oldest minute's weather data
def add_current_data(datalogger):
	print("shifting previous weather data")
	global weather_data_list
	print("size " + str(len(weather_data_list)))

	# Drop off the oldest weather data
	if(len(weather_data_list) > mins_of_previous_data):
		weather_data_list.pop()

	# Format newest data and add to front of weather data list
	cur_data = format_current_weather_data(datalogger)
	print("cur_data: " + str(cur_data))
	weather_data_list.insert(0, cur_data) # should use a deque but maybe will fix later
	print("weather data list: " + str(weather_data_list))

class VerifiedPowerData:
	def __init__(self, dt, perc, pv, av):
		self.verified_time = dt
		self.percentage = perc
		self.predicted_value = pv
		self.actual_value = av

class PowerPredictionRunner(Thread):
	def __init__(self):
		Thread.__init__(self)
		#setup db here
		print("setup db here")

		#creds will need to be created on each system
		self.client = pymongo.MongoClient(creds.base_url + creds.username + creds.separator + creds.password + creds.cluster_url)
		self.db = self.client.CloudTrackingData
		datalogger_connected = False

		# Continue checking for a datalogger connection at the given USB address
		while(not datalogger_connected):
			try:
				print("looking for datalogger")
				self.datalogger = Datalogger('/dev/ttyUSB0') #path will need to change per system
				datalogger_connected = True
				print("datalogger connected")
			except:
				# Wait 10 seconds then check if datalogger has been connected
				time.sleep(10)
				print("datalogger not connected")
			
		# Amount of time to sleep between runs in seconds
		self.sleep_time = 60
		self.predicted_power = 0

		# How many minutes to predict power in the future
		self.predict_out_mins = 15
		self.the_date = datetime.utcnow()
		
	def run(self):
		while(True):
			starttime = time.time()
			self.the_date = datetime.utcnow()

			# Get the current weather data to use for the predictions
			# starting at the current minute
			self.datalogger.poll()
			add_current_data(self.datalogger)
			final_data = Predict.toTimeSeries(weather_data_list, timesteps=3)

			print("final_data: " + str(final_data))

			# Returns a list of length number of minutes to predict
			self.predicted_power = Predict.makePrediction(final_data, self.predict_out_mins)
			print("predicted power")
			print(self.predicted_power)
			vd_list = self.run_verification()
			height = self.calculate_cloud_height()

			# Send various information to the db
			self.send_cloud_height_data_to_db(height)
			self.send_power_prediction_data_to_db(self.predicted_power)
			self.send_weather_data_to_db()
			self.send_verification_data_to_db(vd_list)
			print("starting to sleep")

			# Sleep for 'sleep_time' - when the thread began running
			# Ex: 60 seconds sleep time, 10 seconds execution time, 60 - 10 = 50 seconds sleep time
			time.sleep(self.sleep_time - ((time.time() - starttime) % self.sleep_time))

	#NOTE: Not used currently
	def get_previous_data(self):
		print("getting previous data")
		global weather_data_list
		db_response = self.db.WeatherData.find().sort([('_id', -1)]).limit(5)

		for doc in db_response:
			print(doc)
			temp_list = []
			temp_list.append(doc['slrFD_W'])
			temp_list.append(doc['windDir'])
			temp_list.append(doc['ws_ms'])
			temp_list.append(doc['airT_C'])
			#dt = datetime.strptime(temp_list.append(str(doc['date'])), "%m-%d-%H-%M")
			dt = doc['date']
			temp_list.append(dt.month)
			temp_list.append(dt.day)
			temp_list.append(dt.hour)
			temp_list.append(dt.minute)
			weather_data_list.append(temp_list)
			#print("temp_list len: " + str(len(temp_list)))
			#print(temp_list)

	def send_cloud_height_data_to_db(self, c_height):
		print("sendind cloud data to db")
		the_date = self.the_date
		post = {"author": "power_verification.py",
			"temperature": c_height.temperature,
			"humidity": c_height.humidity,
			"dew_point": c_height.dew_point,
			"cloud_height": c_height.cloud_height,
			"calc_time": c_height.calc_time,
			"calc_time_mins_only": c_height.calc_time_mins_only,
			"date": the_date,
			"date_mins_only": the_date.replace(second=0, microsecond=0),
			"system_num": substation.id}
		
		posts = self.db.CloudHeightData
		post_id = posts.insert_one(post).inserted_id
		print("post_id: " + str(post_id))

	def send_power_prediction_data_to_db(self, predicted_power):
		print("sending predicted power")
		dt = self.the_date.replace(second=0, microsecond=0)
		post = {"author": "power_prediction.py",
				"power_predictions": predicted_power,
				"prediction_start_time": dt,
				# Length of the list will correspond to number of minutes of predictions
				"prediction_end_time": dt + timedelta(minutes=len(predicted_power)),
				"system_num": substation.id}
		
		posts = self.db.PowerPredictionData
		post_id = posts.insert_one(post).inserted_id
		print("post_id: " + str(post_id))

	def send_weather_data_to_db(self):
		print("sendind data to db")
		the_date = self.the_date
		post = {"author": "power_prediction.py",
				"slrFD_W": self.datalogger.weather_data.slrFD_W,
				"rain_mm": self.datalogger.weather_data.rain_mm,
				"strikes": self.datalogger.weather_data.strikes,
				"dist_km": self.datalogger.weather_data.dist_km,
				"ws_ms": self.datalogger.weather_data.ws_ms,
				"windDir": self.datalogger.weather_data.windDir,
				"maxWS_ms": self.datalogger.weather_data.maxWS_ms,
				"airT_C": self.datalogger.weather_data.airT_C,
				"vp_mmHg": self.datalogger.weather_data.vp_mmHg,
				"bp_mmHg": self.datalogger.weather_data.bp_mmHg,
				"rh": self.datalogger.weather_data.rh,
				"rht_c": self.datalogger.weather_data.rht_c,
				"tiltNS_deg": self.datalogger.weather_data.tiltNS_deg,
				"tiltWE_deg": self.datalogger.weather_data.tiltWE_deg,
				"date": self.the_date,
				"date_mins_only": the_date.replace(second=0, microsecond=0),
				"date_time_only": the_date.replace(year=1970, month=1, day=1),
				"system_num": substation.id}
		
		posts = self.db.WeatherData
		post_id = posts.insert_one(post).inserted_id
		print("post_id: " + str(post_id))

	def send_verification_data_to_db(self, vd):
		print("sending verification info to db")
		vd_json = jsonpickle.encode(vd, make_refs=False)
		post = {"author": "power_prediction.py",
				"verified_power_data": vd_json,
				"verified_time": self.the_date,
				"system_num": substation.id}
		
		posts = self.db.PowerVerificationData
		post_id = posts.insert_one(post).inserted_id
		print("post_id: " + str(post_id))

	# Calculate cloud height from air temp and relative humidity
	def calculate_cloud_height(self):
		return CloudHeightData(self.datalogger.weather_data.airT_C, self.datalogger.weather_data.rh)

	# Verifies how close each of the power predictions were for the current minute.
	# Ex: Verifies prediction made at 13:00 for 13:15
	#								  13:01 for 13:15
	#								  ...
	#								  13:14 for 13:15
	def run_verification(self):
		print("running verification")
		prev_preds = None
		prev_power = -1
		min_offset = self.predict_out_mins
		verified_data_list = []

		 # This gets the 15 most recent entries in the power prediction collection.
		db_response = self.db.PowerPredictionData.find({'system_num': substation.id}).sort([('_id', -1)]).limit(self.predict_out_mins)

		# Find returns a 'cursor' to the document, not the actual document so you
		# must iterate the cursor to get the actual documents
		for doc in db_response:
			prev_preds = doc['power_predictions']

			if prev_preds is not None:
				# Get data starting from the beginning of the list
				prev_power_pred = prev_preds[self.predict_out_mins - min_offset]
				min_offset = min_offset - 1

				# Get the real power measured by the weather sensor. Convert to units of the ML model - kw/h
				prev_power_real = (self.datalogger.weather_data.slrFD_W * substation.panel_area / substation.kw) * substation.panel_eff
				#print("real: " + str(prev_power))
				diff = abs(prev_power_real - prev_power_pred)
				error_perc = diff/prev_power_pred * 100
				vd = VerifiedPowerData(self.the_date, error_perc, prev_power_pred, prev_power_real)
				verified_data_list.append(vd)
		
		# This list stores prediction verification from the oldest prediction to latest prediction
		return verified_data_list
		
# Simply poll the datalogger once per minute to get enough data to begin making predictions.
# Likely 15 minutes of data. This should only need to be run when there has been an extended
# amount of time from the last system run.
class Get_Data_On_Startup(Thread):
	def __init__(self, finished_getting_data_event=None):
		Thread.__init__(self)
		print("setting up initial data gathering thread")
		#global weather_data_list
		#weather_data_list.clear()
		self.finished_getting_data_event = finished_getting_data_event
		self.run_num = 1
		self.client = pymongo.MongoClient(creds.base_url + creds.username + creds.separator + creds.password + creds.cluster_url)
		self.db = self.client.cloudTrackingData
		datalogger_connected = False

		# Continue checking for a datalogger connection at the given USB address
		while(not datalogger_connected):
			try:
				print("looking for datalogger")

				# Path will need to change per system
				self.datalogger = Datalogger('/dev/ttyUSB0') #path will need to change per system
				datalogger_connected = True
				print("datalogger connected")
			except:
				# Wait 10 seconds then check if datalogger has been connected
				time.sleep(10)
				print("datalogger not connected")
		
		self.sleep_time = 60 #60 seconds

	# Add the current weather data to the weather data list and send it to the db
	def run(self):
		while(True):
			starttime = time.time()
			print("getting data on start run " + str(self.run_num))
			self.the_date = datetime.utcnow()
			self.datalogger.poll()
			add_current_data(self.datalogger)
			self.send_weather_data_to_db()

			# Got all the data we need
			if self.run_num == mins_of_previous_data:
				break

			self.run_num = self.run_num + 1
			time.sleep(self.sleep_time - ((time.time() - starttime) % self.sleep_time))
		
		print("Got " + str(mins_of_previous_data) + " mins of data")
		self.finished_getting_data_event.set()

# NOTE: will want to refactor to avoid replicating this code maybe
	def send_weather_data_to_db(self):
		print("sendind data to db")
		the_date = self.the_date
		post = {"author": "power_prediction.py",
				"slrFD_W": self.datalogger.weather_data.slrFD_W,
				"rain_mm": self.datalogger.weather_data.rain_mm,
				"strikes": self.datalogger.weather_data.strikes,
				"dist_km": self.datalogger.weather_data.dist_km,
				"ws_ms": self.datalogger.weather_data.ws_ms,
				"windDir": self.datalogger.weather_data.windDir,
				"maxWS_ms": self.datalogger.weather_data.maxWS_ms,
				"airT_C": self.datalogger.weather_data.airT_C,
				"vp_mmHg": self.datalogger.weather_data.vp_mmHg,
				"bp_mmHg": self.datalogger.weather_data.bp_mmHg,
				"rh": self.datalogger.weather_data.rh,
				"rht_c": self.datalogger.weather_data.rht_c,
				"tiltNS_deg": self.datalogger.weather_data.tiltNS_deg,
				"tiltWE_deg": self.datalogger.weather_data.tiltWE_deg,
				"date": self.the_date,
				"date_mins_only": the_date.replace(second=0, microsecond=0),
				"date_time_only": the_date.replace(year=1970, month=1, day=1),
				"system_num": substation.id}
		
		posts = self.db.WeatherData
		post_id = posts.insert_one(post).inserted_id
		print("post_id: " + str(post_id))
		
def main():
	dbg = False
	if dbg:
		global weather_data_list
		weather_data_list =[[0.0, 261.70001220703125, 0.15000000596046448, 24.799999237060547, 9, 28, 21, 53],
		[0.0, 254.10000610351562, 0.05999999865889549, 24.700000762939453, 9, 28, 21, 52],
		[0.0, 237.1999969482422, 0.05999999865889549, 24.700000762939453, 9, 28, 21, 51]]
		print(weather_data_list)
	else:
		finished_getting_data_event = Event()
		get_data_on_startup = Get_Data_On_Startup(finished_getting_data_event)
		get_data_on_startup.start()
		finished_getting_data_event.wait()

	power_prediction_runner = PowerPredictionRunner()
	print("============================")
	print("STARTING MAIN WORKER THREAD")
	print("============================")
	power_prediction_runner.start()
		
if __name__ == "__main__":
	main()
