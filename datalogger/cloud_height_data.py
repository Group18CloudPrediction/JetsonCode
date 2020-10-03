from datetime import datetime

class CloudHeightData:
	def __init__(self, temp, hum):
		self.temperature = temp
		self.humidity = hum
		self.dew_point = temp - ((100 - hum) / 5)
		self.cloud_height = (1000 * (temp - self.dew_point)) / 4.4
		self.calc_time = datetime.utcnow()
		self.calc_time_mins_only = self.calc_time.replace(second=0, microsecond=0)
