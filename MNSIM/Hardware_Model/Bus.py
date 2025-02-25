import configparser as cp

class Bus:
    def __init__(self, SimConfig_path):
        bus_config = cp.ConfigParser()
        bus_config.read(SimConfig_path, encoding='UTF-8')
        self.bus_bitwidth = int(bus_config.get('Bus level', 'Bus_Bitwidth'))
        self.bus_tech = int(bus_config.get('Bus level', 'Bus_Tech'))
        self.bus_choice = int(bus_config.get('Bus level', 'Bus_Choice'))  #不同类型的bus
        

    def calculate_area(self):
        # Implement the logic to calculate the area based on the configuration
        area = 0
        # Example calculation (replace with actual logic)
        
        return area

    def calculate_latency(self):
        # Implement the logic to calculate the latency based on the configuration
        latency = 0
        # Example calculation (replace with actual logic)
        
        return latency

    def calculate_energy(self):
        # Implement the logic to calculate the energy based on the configuration
        energy = 0
        # Example calculation (replace with actual logic)
        
        return energy