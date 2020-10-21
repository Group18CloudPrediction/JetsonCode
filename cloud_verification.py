#cloud location verification
import minimalmodbus as mmbus
import serial
import time
import pymongo
import creds
import config.creds as creds
import config.substation_info as substation

from datalogger.datalogger import Datalogger
from threading import Thread, Event
from datetime import datetime, timedelta

class VerifiedCloudData:
	def __init__(self, dt, perc, pv, av):
		self.verified_time = dt
		self.percentage = perc
		self.predicted_value = pv
		self.actual_value = av
		
class CloudLocationRunner(Thread):
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
		self.predicted_cloud_location = False
		self.the_date = datetime.utcnow()

		
	def run(self):
		while(True):
			starttime = time.time()
			self.the_date = datetime.utcnow()
			self.datalogger.poll()
			self.predicted_cloud_location = self.get_cloud_location_from_db()
			res = self.run_verification()
			av = self.datalogger.weather_data.slrFD_W
			vd = VerifiedCloudData(self.the_date, res, self.predicted_cloud_location, av)
			self.send_verification_data_to_db(vd)
			print("starting to sleep")
			time.sleep(self.sleep_time - ((time.time() - starttime) % self.sleep_time))
            
	def get_cloud_location_from_db(self):
		db_response = self.db.CloudLocationPredictions.find().sort([('_id', -1)]).limit(15)

		for doc in db_response:
			pass
			
	def send_verification_data_to_db(self, vd):
		print("sending verification info to db")
		post = {"author": "cloud_verification.py",
				"percentage_error": vd.percentage,
				"predicted_cloud": vd.predicted_value,
				"actual_cloud": vd.actual_value,
				"verified_time": vd.verified_time}
		
		posts = self.db.CloudVerificationData
		post_id = posts.insert_one(post).inserted_id
		print("post_id: " + str(post_id))

	def run_verification(self):
		print("running verification")
		
def main():
	cloud_location_runner = CloudLocationRunner()
	cloud_location_runner.start()
		
if __name__ == "__main__":
	main()