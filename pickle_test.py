import pymongo
import config.creds as creds
import pickle

class VerifiedData:
	def __init__(self, dt, perc, pv, av):
		self.verified_time = str(dt)
		self.percentage = str(perc)
		self.predicted_value = pv
		self.actual_value = av

client = pymongo.MongoClient("mongodb+srv://" + creds.username + ":" + creds.password + "@cluster0.lgezy.mongodb.net/<dbname>?retryWrites=true&w=majority")
db = client.cloudTrackingData

db_response = db.PowerVerificationData.find().sort([('_id', -1)]).limit(1)

for doc in db_response:
	print(doc)
	author = doc['author']
	preds_list = pickle.loads(doc['verified_power_data'])
	time = doc["verified_time"]
	system = doc['system_num']
	
	for pred in preds_list:
		print(pred.verified_time)
		print(pred.percentage)
		print(pred.predicted_value)
		print(pred.actual_value)
