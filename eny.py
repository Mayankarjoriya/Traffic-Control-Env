import random

class TrafficEnv:


    def state(self):
        self.state = {
            "north_cars": random.randint(1, 20),  # 0-20 random cars
            "south_cars": random.randint(1,20),
            "west_cars": random.randint(1,20),
            "east_cars": random.randint(1,20),
            "north_wait": 0,
            "west_wait": 0,
            "east_wait": 0,
            "south_wait": 0,
            "current_green_0": None,  # sab red matlab "none"
            "current_green_1": None,
            "ambulance": False
        }

        return self.state
    
    def step(self, action):
        # Setp 1 selected lane turn green
        lanes = ["west", "north", "south", "east"]
        # self.state["current_green"] = lanes[action]

        if action == "NS_GREEN":
            self.state["current_green_0"] = "north"
            self.state["current_green_1"] = "south" 
            
            # Step 2 Reward
            Reward = 0.0
            Reward -= self.state["east_cars"]*0.1
            Reward -= self.state["west_cars"]*0.1

            # Measure waiting Time
            self.state["east_wait"] += 10
            self.state['west_wait'] += 10

            # Empty no. of cars in selected lane
            self.state["north_cars"] = 0
            self.state["south_cars"] = 0

            

        elif action == "EW_GREEN":
            self.state["current_green_0"] = "east"
            self.state["current_green_1"] = "west" 

            # Step 2 Reward
            Reward = 0.0
            Reward -= self.state["north_cars"]*0.1
            Reward -= self.state["south_cars"]*0.1
           

            # Measure waiting Time
            self.state["south_wait"] += 10
            self.state['north_wait'] += 10

            # Empty no. of cars in selected lane
            self.state["east_cars"] = 0
            self.state["west_cars"] = 0

        

        return self.state , Reward


    def reset(self, task_id=1):
        # Easy
        if task_id == 1:
            self.state = {
            "north_cars": random.randint(1, 10),  # 0-20 random cars
            "south_cars": random.randint(1,10),
            "east_cars": random.randint(1,10),
            "west_cars": random.randint(1,10),
            "north_wait": 0,
            "west_wait": 0,
            "east_wait": 0,
            "south_wait": 0,
            "current_green_0": None,  # sab red matlab "none"
            "current_green_1": None,
            "ambulance": False
        }
            
        # medium
        elif task_id == 2:


        # Hard
        elif task_id == 3:  
            self.state = {
            "north_cars": random.randint(1, 30),  # 0-20 random cars
            "south_cars": random.randint(1,30),
            "east_cars": random.randint(1,30),
            "west_cars": random.randint(1,30),
            "north_wait": 0,
            "west_wait": 0,
            "east_wait": 0,
            "south_wait": 0,
            "current_green_0": None,  # sab red matlab "none"
            "current_green_1": None,
            "ambulance": "north"
        }
       
