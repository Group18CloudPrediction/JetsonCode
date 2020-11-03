#NOTE: This will not be used but will be left and commented
# in case future users want a script that ONLY polls the datalogger
# for weather data, calculates cloud height, and sends both to the db.

import time
from datetime import datetime
from threading import Thread, Event
import pymongo
import config.creds as creds
import config.substation_info as substation
from datalogger.datalogger import Datalogger
from datalogger.cloud_height_data import CloudHeightData

class DataloggerThread(Thread):
	def __init__(self):
		Thread.__init__(self)

		# Creds will need to be created on each system
		self.client = pymongo.MongoClient(creds.base_url + creds.username + creds.separator + creds.password + creds.cluster_url)
		self.db = self.client.cloudTrackingData
		datalogger_connected = False

		while(not datalogger_connected):
			try:
				print("looking for datalogger")
				# Path will need to change per system
				self.datalogger = Datalogger('/dev/ttyS5')
				datalogger_connected = True
				print("datalogger connected")
			except:
				# Wait 10 seconds then check if datalogger has been connected
				time.sleep(10)
				print("datalogger not connected")
			
		self.sleep_time = 60

	def run(self):
		while(True):
			starttime = time.time()
			self.the_date = datetime.utcnow()

			# Get the current weather data
			self.datalogger.poll()
			height = self.calculate_cloud_height()
			self.send_weather_data_to_db()
			self.send_cloud_height_data_to_db(height)
			time.sleep(self.sleep_time - ((time.time() - starttime) % self.sleep_time))

	def send_weather_data_to_db(self):
		print("sendind data to db")
		the_date = self.the_date
		post = {"author": "datalogger_runner.py",
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
				"system_num": substation.id}
		
		posts = self.db.WeatherData_Only
		post_id = posts.insert_one(post).inserted_id
		print("post_id: " + str(post_id))

	def send_cloud_height_data_to_db(self, c_height):
		print("sendind cloud data to db")
		the_date = self.the_date
		post = {"author": "datalogger_runner.py",
			"temperature": c_height.temperature,
			"humidity": c_height.humidity,
			"dew_point": c_height.dew_point,
			"cloud_height": c_height.cloud_height,
			"calc_time": c_height.calc_time,
			"calc_time_mins_only": c_height.calc_time_mins_only,
			"date": self.the_date,
			"date_mins_only": the_date.replace(second=0, microsecond=0),
			"system_num": substation.id}
		
		posts = self.db.CloudHeightData
		post_id = posts.insert_one(post).inserted_id
		print("post_id: " + str(post_id))

	def calculate_cloud_height(self):
		return CloudHeightData(self.datalogger.weather_data.airT_C, self.datalogger.weather_data.rh)

def main():
	datalogger_runner = DataloggerThread()
	datalogger_runner.start()
		
if __name__ == "__main__":
	main()
