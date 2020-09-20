import minimalmodbus as mmbus
import serial
import time
import pymongo
import creds
import Predict
import Train
from threading import Thread, Event
from datetime import datetime

class WeatherData:
	def __init__(self, slrFD_W, rain_mm, strikes, dist_km, ws_ms, windDir, maxWS_ms, airT_C, vp_mmHg, bp_mmHg, rh, rht_c, tiltNS_deg, tiltWE_deg):
		self.slrFD_W = slrFD_W #1
		self.rain_mm = rain_mm
		self.strikes = strikes
		self.dist_km = dist_km
		self.ws_ms = ws_ms #3
		self.windDir = windDir #2
		self.maxWS_ms = maxWS_ms
		self.airT_C = airT_C #4
		self.vp_mmHg = vp_mmHg
		self.bp_mmHg = bp_mmHg
		self.rh = rh
		self.rht_c = rht_c
		self.tiltNS_deg = tiltNS_deg
		self.tiltWE_deg = tiltWE_deg
		self.poll_time = datetime.utcnow()
		# mm, dd, hh, mm #5
		# [slrFD_W, windDir, ws_ms, airT_C, time]

class VerifiedData:
	def __init__(self, dt, perc, pv, av):
		self.verified_time = str(dt)
		self.percentage = str(perc)
		self.predicted_value = pv
		self.actual_value = av
		
class Datalogger:
	def __init__(self, path):
		self.ins = mmbus.Instrument(path, 1, mode='rtu')
		self.ins.serial.baudrate = 9600
		self.ins.serial.bytesize = 8
		self.ins.serial.parity = serial.PARITY_NONE
		self.ins.serial.stopbits = 1
		self.ins.serial.timeout = 2
		
	def poll(self):
		slrFD_W = self.ins.read_float(0, functioncode=3, number_of_registers=2, byteorder=0)
		rain_mm = self.ins.read_float(2, functioncode=3, number_of_registers=2, byteorder=0)
		strikes = self.ins.read_float(4, functioncode=3, number_of_registers=2, byteorder=0)
		dist_km = self.ins.read_float(6, functioncode=3, number_of_registers=2, byteorder=0)
		ws_ms = self.ins.read_float(8, functioncode=3, number_of_registers=2, byteorder=0)
		windDir = self.ins.read_float(10, functioncode=3, number_of_registers=2, byteorder=0)
		maxWS_ms = self.ins.read_float(12, functioncode=3, number_of_registers=2, byteorder=0)
		airT_C = self.ins.read_float(14, functioncode=3, number_of_registers=2, byteorder=0)
		vp_mmHg = self.ins.read_float(16, functioncode=3, number_of_registers=2, byteorder=0)
		bp_mmHg = self.ins.read_float(18, functioncode=3, number_of_registers=2, byteorder=0)
		rh = self.ins.read_float(20, functioncode=3, number_of_registers=2, byteorder=0)
		rht_c = self.ins.read_float(22, functioncode=3, number_of_registers=2, byteorder=0)
		tiltNS_deg = self.ins.read_float(24, functioncode=3, number_of_registers=2, byteorder=0)
		tiltWE_deg = self.ins.read_float(26, functioncode=3, number_of_registers=2, byteorder=0)
		
		self.weather_data = WeatherData(slrFD_W, rain_mm, strikes, dist_km, ws_ms, windDir, maxWS_ms, airT_C, vp_mmHg, bp_mmHg, rh, rht_c, tiltNS_deg, tiltWE_deg)
		self.print_data_test()
		
	def print_data_test(self):
		print(str(self.weather_data.slrFD_W))
		print(str(self.weather_data.rain_mm))
		print(str(self.weather_data.strikes))
		print(str(self.weather_data.dist_km))
		print(str(self.weather_data.ws_ms))
		print(str(self.weather_data.windDir))
		print(str(self.weather_data.maxWS_ms))
		print(str(self.weather_data.airT_C))
		print(str(self.weather_data.vp_mmHg))
		print(str(self.weather_data.bp_mmHg))
		print(str(self.weather_data.rh))
		print(str(self.weather_data.rht_c))
		print(str(self.weather_data.tiltNS_deg))
		print(str(self.weather_data.tiltWE_deg))
				
class WeatherDataDBRunner(Thread):
	def __init__(self):
		Thread.__init__(self)
		#setup db here
		print("setup db here")

		#creds will need to be created on each system
		self.client = pymongo.MongoClient("mongodb+srv://" + creds.username + ":" + creds.password + "@cluster0.lgezy.mongodb.net/<dbname>?retryWrites=true&w=majority")
		self.db = self.client.cloudTrackingData
		self.datalogger = Datalogger('/dev/ttyS5') #path will need to change per system
		self.sleep_time = 60 #60 seconds
		self.predicted_power = 0
		self.the_date = datetime.utcnow()
		self.weather_data = [] # includes 4 past weather datas and current weather data
		self.get_previous_data()

		
	def run(self):
		while(True):
			self.the_date = datetime.utcnow()
			self.datalogger.poll() # get the current weather data to use for the next prediction
			self.add_current_data()
			#print("weather data before to timeseries")
			#print(self.weather_data)
			final_data = Train.toTimeSeries(self.weather_data, timesteps=3)
			#print("final_data: " + str(final_data))
			self.predicted_power = Predict.makePrediction(final_data, 15)
			print("predicted power")
			print(self.predicted_power)
			perc = self.run_verification()
			av = self.datalogger.weather_data.slrFD_W
			vd = VerifiedData(self.the_date, perc, self.predicted_power, av)
			# self.send_power_prediction_data_to_db(predicted_power)
			self.send_weather_data_to_db()
			self.send_verification_data_to_db(vd)
			time.sleep(self.sleep_time)

	def get_previous_data(self):
		print("getting previous data")

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
			self.weather_data.append(temp_list)
			#print("temp_list len: " + str(len(temp_list)))
			#print(temp_list)

	def add_current_data(self):
		print("shifting previous weather data")
		self.weather_data.pop()
		cur_data = self.format_current_weather_data()
		print("cur_data: " + str(cur_data))
		self.weather_data.insert(0, cur_data) # should use a deque but maybe will fix later

	def format_current_weather_data(self):
		temp = []
		temp.append(self.datalogger.weather_data.slrFD_W)
		temp.append(self.datalogger.weather_data.windDir)
		temp.append(self.datalogger.weather_data.ws_ms)
		temp.append(self.datalogger.weather_data.airT_C)
		temp.append(self.datalogger.weather_data.poll_time.month)
		temp.append(self.datalogger.weather_data.poll_time.day)
		temp.append(self.datalogger.weather_data.poll_time.hour)
		temp.append(self.datalogger.weather_data.poll_time.minute)

		return temp
			
	def send_power_prediction_data_to_db(self, predicted_power):
		print("sending predicted power")

	def send_weather_data_to_db(self):
		print("sendind data to db")
		the_date = self.the_date
		post = {"author": "datalogger",
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
				"tags": ["weather_data", "datalogger", "weather", "weather_station", "verified_data"],
				"date": self.the_date,
				"date_mins_only": the_date.replace(second=0, microsecond=0),
				"system_num": "PLACEHOLDER_REPLACE"}
		
		posts = self.db.WeatherData
		post_id = posts.insert_one(post).inserted_id
		print("post_id: " + str(post_id))

	def send_verification_data_to_db(self, vd):
		print("sending verification info to db")
		post = {"author": "datalogger",
				"percentage_error": vd.percentage,
				"predicted_power": vd.predicted_value,
				"actual_power": vd.actual_value,
				"verified_time": vd.verified_time}
		
		posts = self.db.PowerVerificationData
		post_id = posts.insert_one(post).inserted_id
		print("post_id: " + str(post_id))

	def run_verification(self):
		print("running verification")
		real = self.datalogger.weather_data.slrFD_W
		diff = abs(self.predicted_power - real)
		error_perc = diff/real
		return error_perc * 100

class Get_Data_On_Startup(Thread):
	def __init__(self, finished_getting_data_event=None):
		Thread.__init__(self)
		print("setting up initial data gathering thread")
		self.finished_getting_data_event = finished_getting_data_event
		self.run_num = 0
		self.mins_of_data = 1
		self.client = pymongo.MongoClient("mongodb+srv://" + creds.username + ":" + creds.password + "@cluster0.lgezy.mongodb.net/<dbname>?retryWrites=true&w=majority")
		self.db = self.client.cloudTrackingData
		self.datalogger = Datalogger('/dev/ttyS5') #path will need to change per system
		self.sleep_time = 60 #60 seconds


	def run(self):
		while(self.run_num < self.mins_of_data):
			print("getting data on start run" + str(self.run_num))
			self.the_date = datetime.utcnow()
			self.datalogger.poll()
			self.send_weather_data_to_db()
			self.run_num = self.run_num + 1
			time.sleep(self.sleep_time)
		
		print("Got 5 mins of data")
		self.finished_getting_data_event.set()

# NOTE: will want to refactor to avoid replicating this code
	def send_weather_data_to_db(self):
		print("sendind data to db")
		the_date = self.the_date
		post = {"author": "datalogger",
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
				"tags": ["weather_data", "datalogger", "weather", "weather_station", "verified_data"],
				"date": self.the_date,
				"date_mins_only": the_date.replace(second=0, microsecond=0),
				"system_num": "PLACEHOLDER_REPLACE"}
		
		posts = self.db.WeatherData
		post_id = posts.insert_one(post).inserted_id
		print("post_id: " + str(post_id))


		
def main():
	finished_getting_data_event = Event()
	weather_data_db_runner = WeatherDataDBRunner()
	#get_data_on_startup = Get_Data_On_Startup(finished_getting_data_event)
	#get_data_on_startup.start()
	#finished_getting_data_event.wait()
	print("============================")
	print("STARTING MAIN WORKER THREAD")
	print("============================")
	weather_data_db_runner.start()
		
if __name__ == "__main__":
	main()

# Get the instrument from the PC port it's plugged into
# and the modbus device address. Set mode to rtu.
#ins = mmbus.Instrument('/dev/ttyS5', 1, mode='rtu') 

# Just set some properties to match the datalogger
#ins.debug = True
#ins.serial.baudrate = 9600
#ins.serial.bytesize = 8
#ins.serial.parity = serial.PARITY_NONE
#ins.serial.stopbits = 1
#ins.serial.timeout = 2
#print(ins)

# Read 4 registers starting at register 1. There are 2 floats being
# stored, so they use 4 registers in total.
#data = ins.read_registers(1, number_of_registers=2, functioncode=3)
#print('data: ' + str(data))

# read a float starting at register 7. a float uses 2 registers
# the first arguments to read_float is the register number to read
#fl = ins.read_float(12, functioncode=3, number_of_registers=2, byteorder=0)
#print('float ' + str(round(fl, 2)))

