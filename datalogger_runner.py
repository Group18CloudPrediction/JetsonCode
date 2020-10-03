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
		#setup db here
		print("setup db here")

		#creds will need to be created on each system
		self.client = pymongo.MongoClient("mongodb+srv://" + creds.username + ":" + creds.password + "@cluster0.lgezy.mongodb.net/<dbname>?retryWrites=true&w=majority")
		self.db = self.client.cloudTrackingData
		datalogger_connected = False

		while(not datalogger_connected):
			try:
				print("looking for datalogger")
				self.datalogger = Datalogger('/dev/ttyS5') #path will need to change per system
				datalogger_connected = True
				print("datalogger connected")
			except:
				time.sleep(10) # wait 10 seconds then check if datalogger has been connected
				print("datalogger not connected")
			
		self.sleep_time = 60 #60 seconds

	def run(self):
		while(True):
			starttime = time.time()
			self.the_date = datetime.utcnow()
			self.datalogger.poll() # get the current weather data
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
				"tags": ["weather_data", "datalogger", "weather", "weather_station", "verified_data"],
				"date": self.the_date,
				"date_mins_only": the_date.replace(second=0, microsecond=0),
				"system_num": substation.id}
		
		posts = self.db.WeatherData
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
			"tags": ["cloud_data", "datalogger", "weather", "weather_station"],
			"date": self.the_date,
			"date_mins_only": the_date.replace(second=0, microsecond=0),
			"system_num": substation.id}
		
		posts = self.db.CloudHeightData
		post_id = posts.insert_one(post).inserted_id
		print("post_id: " + str(post_id))


	def calculate_cloud_height(self):
		c_height = CloudHeightData(self.datalogger.weather_data.airT_C, self.datalogger.weather_data.rh)

		return c_height

def main():
	datalogger_runner = DataloggerThread()
	datalogger_runner.start()
		
if __name__ == "__main__":
	main()
