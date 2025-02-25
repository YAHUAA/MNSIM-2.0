#!/usr/bin/python
#-*-coding:utf-8-*-
import configparser as cp
import os
import math
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
# Default SimConfig file path: MNSIM_Python/SimConfig.ini


class device(object):
	def __init__(self, SimConfig_path,device_type):
		device_config = cp.ConfigParser()
		device_config.read(SimConfig_path, encoding='UTF-8')
		self.device_type = device_type

		self.device_tech = float(device_config.get(self.device_type, 'Device_Tech'))
		#self.device_type = device_config.get(self.device_type, 'Device_Type')
		self.device_area = float(device_config.get(self.device_type, 'Device_Area'))
		self.device_read_voltage_level = int(device_config.get(self.device_type, 'Read_Level'))
		assert self.device_read_voltage_level >= 0, "Read voltage level < 0"
		self.device_read_voltage = list(map(float, device_config.get(self.device_type, 'Read_Voltage').split(',')))
		assert self.device_read_voltage_level == len(self.device_read_voltage), "Read voltage setting error"

		self.device_write_voltage_level = int(device_config.get(self.device_type, 'Write_Level'))
		assert self.device_write_voltage_level >= 0, "Write voltage level < 0"
		self.device_write_voltage = list(map(float, device_config.get(self.device_type, 'Write_Voltage').split(',')))
		assert self.device_write_voltage_level == len(self.device_write_voltage), "Write voltage setting error"

		#-----------------#
		self.device_read_latency = float(device_config.get(self.device_type, 'Read_Latency'))
		self.device_write_latency = float(device_config.get(self.device_type, 'Write_Latency'))

		self.device_level = int(device_config.get(self.device_type, 'Device_Level'))
		assert self.device_level >= 0, "DCIM resistance level < 0"
		self.device_resistance = list(map(float, device_config.get(self.device_type, 'Device_Resistance').split(',')))
		assert self.device_level == len(self.device_resistance), "DCIM resistance setting error"

		self.device_variation = float(device_config.get(self.device_type, 'Device_Variation'))

		self.device_read_energy = float(device_config.get(self.device_type, 'Read_Energy'))
		self.device_write_energy = float(device_config.get(self.device_type, 'Write_Energy'))
		self.device_read_power = self.device_read_energy/self.device_read_latency
		self.device_write_power = self.device_write_energy/self.device_write_latency
		

	def calculate_device_read_power(self, R = None, V = None):
		# R is the resistance of memristor, None means use default resistance (Sqrt(R_on*R_off))
		if R is None:
			# R = math.sqrt(float(self.device_resistance[0])*float(self.device_resistance[-1]))
			# R = float(self.device_resistance[-1]) #worst case estimation
			# R = 0.75*float(self.device_resistance[0]) + 0.25*float(self.device_resistance[-1])
			R = (float(self.device_resistance[0])*float(self.device_resistance[-1]))/\
				(float(self.device_resistance[-1])*0.67+float(self.device_resistance[0])*0.33)
		assert R > 0, "Resistance <= 0"
		if V is None:
			# V = math.sqrt((self.device_read_voltage[0]**2 + self.device_read_voltage[-1]**2)/2)
			# V = self.device_read_voltage[-1] #worst case estimation
			V = math.sqrt(0.9*(self.device_read_voltage[0]**2) + 0.1*(self.device_read_voltage[-1]**2))
		assert V >= 0, "Voltage < 0"
		self.device_read_power = V ** 2 / R

	def calculate_device_write_power(self, R = None, V = None):
		# only used for NVM
		# R is the resistance of memristor, None means use default resistance (Sqrt(R_on*R_off))
		assert self.type == "NVM", "only the NVM device write power needs to be calculated"
		if R is None:
			R = math.sqrt(float(self.device_resistance[0])*float(self.device_resistance[-1]))
		assert R > 0, "Resistance <= 0"
		if V is None:
			V = (self.device_write_voltage[0] + self.device_write_voltage[-1])/2
		assert V >= 0, "Voltage < 0"
		self.device_write_power = V ** 2 / R

	def device_output(self):
		print("device_tech:", self.device_tech, "nm")
		print("read_voltage:", self.device_read_voltage, "V")
		print("write_voltage:", self.device_write_voltage, "V")
		print("read_latency:", self.device_read_latency, "ns")
		print("write_latency:", self.device_write_latency, "ns")
		print("device_level", self.device_level)
		print("device_resistance:", self.device_resistance, "(ohm)")
		print("device_variation:", self.device_variation, "%")
		print("device_read_power:", self.device_read_power, "W")
		print("device_write_power:", self.device_write_power, "W")

	
def device_test():
	print("load file:",test_SimConfig_path)
	_device = device(test_SimConfig_path,"DCIM")
	_device.calculate_device_read_power()
	_device.calculate_device_write_power()
	_device.device_output()

if __name__ == '__main__':
	device_test()