import random

class TrafficEnv:


    def get_state(self):

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

        elif action == "NE_GREEN":
            self.state["current_green_0"] = "east"
            self.state["current_green_1"] = "north" 

            # Step 2 Reward
            Reward = 0.0
            Reward -= self.state["west_cars"]*0.1
            Reward -= self.state["south_cars"]*0.1
           

            # Measure waiting Time
            self.state["west_wait"] += 10
            self.state['south_wait'] += 10

            # Empty no. of cars in selected lane
            self.state["east_cars"] = 0
            self.state["north_cars"] = 0
        
        elif action == "NW_GREEN":
            self.state["current_green_0"] = "north"
            self.state["current_green_1"] = "west" 

            # Step 2 Reward
            Reward = 0.0
            Reward -= self.state["east_cars"]*0.1
            Reward -= self.state["south_cars"]*0.1
           

            # Measure waiting Time
            self.state["south_wait"] += 10
            self.state['east_wait'] += 10

            # Empty no. of cars in selected lane
            self.state["north_cars"] = 0
            self.state["west_cars"] = 0


        task_id_no = self.state["task_id"]

        if task_id_no == 1:
            done = (
            self.state["north_cars"] == 0 and
            self.state["south_cars"] == 0 and
            self.state["east_cars"] == 0 and
            self.state["west_cars"] == 0
            )
        elif task_id_no == 2:
            done = (
            self.state["north_cars"] < 5 and
            self.state["south_cars"] < 5 and
            self.state["east_cars"]  < 5 and
            self.state["west_cars"]  < 5
            )
        elif task_id_no == 3:
            done = (
            self.state["north_cars"] < 15 and
            self.state["south_cars"] < 15 and
            self.state["east_cars"]  < 15 and
            self.state["west_cars"]  < 15 
            
        #     self.state["ambulance_lane"] in [
        #     self.state["current_green_0"],
        #     self.state["current_green_1"]
        # ]
            )
        else:
            print(f"No Task Assign : {task_id_no}")

 
        info = {
            "total_wait": self.state["north_wait"] + self.state["south_wait"] + self.state["east_wait"] + self.state["west_wait"]
                  
            }
        if self.state["rush_hour"] and self.state["rush_hour"] != False:
            rush_lane= self.state["rush_hour"]
            self.state[f"{rush_lane}_cars"] += random.randint(1,5)
        # Ambulance penalty
        if self.state["ambulance"] and self.state["ambulance_lane"] not in [self.state["current_green_0"], self.state["current_green_1"]]:
            Reward -= 5.0  # Max penalty!

        return self.state , Reward, done, info


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
            "ambulance": False,
            "ambulance_lane": None,
            "rush_hour": False,
            "task_id": 1,
        }
            
        # medium
        elif task_id == 2:
            self.state = {
            "north_cars": random.randint(10, 20),  # 0-20 random cars
            "south_cars": random.randint(10,20),
            "east_cars": random.randint(10,20),
            "west_cars": random.randint(10,20),
            "north_wait": 0,
            "west_wait": 0,
            "east_wait": 0,
            "south_wait": 0,
            "current_green_0": None,  # sab red matlab "none"
            "current_green_1": None,
            "ambulance": False,
            "ambulance_lane": None,
            "rush_hour": "north",
            "task_id":2,
        }



        # Hard
        elif task_id == 3:  
            self.state = {
            "north_cars": random.randint(15, 30),  # 0-20 random cars
            "south_cars": random.randint(15,30),
            "east_cars": random.randint(15,30),
            "west_cars": random.randint(15,30),
            "north_wait": 0,
            "west_wait": 0,
            "east_wait": 0,
            "south_wait": 0,
            "current_green_0": None,  # sab red matlab "none"
            "current_green_1": None,
            "ambulance": True,
            "ambulance_lane": "west",
            "rush_hour": "north",
            "task_id":3,
        }

        return self.state
       
