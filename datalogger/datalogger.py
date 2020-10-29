import minimalmodbus as mmbus
import serial
from datalogger.weather_data import WeatherData

class Datalogger:
	def __init__(self, path):
		print("datalogger constructor")
		self.ins = mmbus.Instrument(path, 1, mode='rtu', debug=True)
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
		return self.weather_data

	def poll_and_return(self):
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
		
		weather_data = WeatherData(slrFD_W, rain_mm, strikes, dist_km, ws_ms, windDir, maxWS_ms, airT_C, vp_mmHg, bp_mmHg, rh, rht_c, tiltNS_deg, tiltWE_deg)
		return weather_data

		
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