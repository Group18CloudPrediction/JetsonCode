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