import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env 
import numpy as np, pandas as pd
from pathlib import Path

class BikeShareEnv(gym.Env):
    """One agent action = choose a station and transfer signed bikes.
    Positive transfer unloads truck to station; negative transfer loads truck.
    Customer OD events occur during travel and service time.
    """
    metadata={"render_modes":[]}
    def __init__(self,stations,od,truck_capacity=30,episode_hours=24,
                 speed_kmh=18,transfer_levels=(-10,-5,0,5,10),
                 demand_scale=1.0,decision_minutes=15,seed=1508):
        """
        Loading inital data for the environment like stations, truck capacity, episode hours,
        average speed, etc. For now we can only unload or load upto 10 bikes and the truck carries 30.

        """
        super().__init__()
        self.stations_df = stations.reset_index(drop=True).copy()
        self.od_demand_df = od.copy()
        self.num_stations = len(self.stations_df)
        
        # Physical system attributes 
        self.cap = self.stations_df["capacity"].to_numpy(float)
        self.station_coordinates = self.stations_df[["lat", "lon"]].to_numpy(float)
        self.station_ids = self.stations_df["station_id"].astype(str).tolist()
        
        self.truck_capacity = truck_capacity
        self.episode_hours = episode_hours
        self.truck_speed_kmh = speed_kmh
        self.transfer_options = np.array(transfer_levels)
        self.demand_scale = demand_scale
        self.decision_minutes = decision_minutes
        self.action_space = spaces.MultiDiscrete([self.num_stations, len(self.transfer_options)])
        
        # Compute exact state vector footprint size
        # (Inventory, Outflow signal, Inflow signal, Upcoming hourly trends, Truck Position One-Hot, Truck Load, Cyclical Time components)
        observation_dimension = (4 * self.num_stations) + self.num_stations + 4
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (observation_dimension,), dtype=np.float32
        )
        self.rng = np.random.default_rng(seed)
        self._build_profiles()

    def _build_profiles(self):
        """
        For every weekday, hour, station we make 2 vectors (hourly_origin_rates and 
        hourly_destination_rates) from the historical data which contains how many customers use that
        station at that time (can be pickup or drop off).
        ex: [0,5,7002]=30 from origin rates means Monday at 5am at the St.George station, 30 people
        picked up a bike and if [0,5,7002]=20 from destination_rates, than 20 people dropped off a bike.
        Ofc we don't use 7002, we use station indexing so make sure to convert the index to the 
        station id to map back to actual station.
        """
        self.hourly_origin_rates = np.zeros((7, 24, self.num_stations))
        self.hourly_destination_rates = np.zeros_like(self.hourly_origin_rates)
        
        station_id_to_index = {station_id: idx for idx, station_id in enumerate(self.station_ids)}
        
        for row in self.od_demand_df.itertuples():
            if row.start_station_id in station_id_to_index and row.end_station_id in station_id_to_index:
                origin_idx = station_id_to_index[row.start_station_id]
                dest_idx = station_id_to_index[row.end_station_id]
                self.hourly_origin_rates[int(row.weekday), int(row.hour), origin_idx] += row.count
                self.hourly_destination_rates[int(row.weekday), int(row.hour), dest_idx] += row.count
                
        self.hourly_origin_rates *= self.demand_scale
        self.hourly_destination_rates *= self.demand_scale

    def _dist(self,a,b):
        """
        Calculating the distance between the two stations a and b using Haversine km
        which isn't that great since it doesnt use the streets and just the long and latitude and then the 
        distance between the two via a direct line. Maybe change this to something else later on?
        """
        # Haversine km
        p1=np.radians(self.station_coordinates[a])
        p2=np.radians(self.station_coordinates[b])
        dist=p2-p1
        h=np.sin(dist[0]/2)**2+np.cos(p1[0])*np.cos(p2[0])*np.sin(dist[1]/2)**2
        return 6371*2*np.arcsin(np.sqrt(h))
    def _obs(self):
        """
        Building out the state observation array with the required vals
        for the RL tracking step. Also returns what we're expecting in the next hour in terms of
        demand. These are then scaled accordingly and returned.
        """
        current_weekday = (self.start_weekday + int(self.current_time_hours // 24)) % 7
        current_hour = int(self.current_time_hours) % 24
        next_hour = (current_hour + 1) % 24
        origin_demand = self.hourly_origin_rates[current_weekday, current_hour]
        dest_demand = self.hourly_destination_rates[current_weekday, current_hour]

        next_origin_demand = self.hourly_origin_rates[current_weekday, next_hour]
        next_dest_demand = self.hourly_destination_rates[current_weekday, next_hour]

        truck_location_one_hot = np.zeros(self.num_stations)
        truck_location_one_hot[self.truck_location_idx] = 1
        
        # Prevent zero-division occurrences with safety scaling
        scaled_origin = origin_demand / self._scale(origin_demand)
        scaled_destination = dest_demand / self._scale(dest_demand)
        combined_future_demand = next_origin_demand + next_dest_demand
        scaled_future_demand = combined_future_demand / self._scale(combined_future_demand)
        
        return np.concatenate([
            self.station_inventories / self.cap,
            scaled_origin,
            scaled_destination,
            scaled_future_demand,
            truck_location_one_hot,
            [self.current_truck_load / self.truck_capacity],
            [np.sin(2 * np.pi * (self.current_time_hours % 24) / 24)],
            [np.cos(2 * np.pi * (self.current_time_hours % 24) / 24)],
            [self.current_time_hours / (self.episode_hours * 24)]]).astype(np.float32)
    
    @staticmethod
    def _scale(x): 
        return max(float(np.max(x)) if len(x) else 0,1.0)

    def get_formatted_time(self) -> str:
        """Returns the current simulation weekday name and clock time string."""
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        current_weekday_idx = (self.start_weekday + int(self.current_time_hours // 24)) % 7
        total_day_hours = self.current_time_hours % 24
        hours = int(total_day_hours)
        minutes = int((total_day_hours - hours) * 60)
        weekday_name = days_of_week[current_weekday_idx]
        clock_time = f"{hours:02d}:{minutes:02d}"
        return f"{weekday_name} at {clock_time}"

    def _customer_events(self,minutes):
        """
        Simulates complete customer trips taking place across the network during the time that passed 
        while the truck was doing an action. Uses an exponential distribution to model the time between 
        sequential trips (Poisson process). For each trip event, an origin and destination station
        are chosen based on historical probabilities.
        - If the origin has a bike AND the destination has an empty dock: The trip succeeds (reward += 1.0)
        - If the origin is empty OR the destination is full: The entire trip fails (reward -= 2.0)
        """
        step_reward = 0.0
        completed_trips = 0
        unmet_requests = 0
        remaining_minutes = minutes
        station_id_to_idx = {station_id: idx for idx, station_id in enumerate(self.station_ids)}
        
        while remaining_minutes > 0:
            current_weekday = (self.start_weekday + int(self.current_time_hours // 24)) % 7
            current_hour = int(self.current_time_hours) % 24
            system_trip_rate = self.hourly_origin_rates[current_weekday, current_hour].sum()
            if system_trip_rate <= 0:
                break
            time_to_next_trip = self.rng.exponential(60 / max(system_trip_rate, 1e-6))
            if time_to_next_trip > remaining_minutes:
                break
            self.current_time_hours += time_to_next_trip / 60
            remaining_minutes -= time_to_next_trip
            # Select an origin station based on demand probabilities
            origin_probs = self.hourly_origin_rates[current_weekday, current_hour]
            origin_station_idx = int(self.rng.choice(self.num_stations, p=origin_probs / origin_probs.sum()))

            # Filter trip data for destination probabilities
            matching_trips = self.od_demand_df[
                (self.od_demand_df.weekday == current_weekday) & 
                (self.od_demand_df.hour == current_hour) & 
                (self.od_demand_df.start_station_id == self.station_ids[origin_station_idx])
            ]
            if matching_trips.empty:
                continue
            destination_probabilities = np.zeros(self.num_stations)
            for row in matching_trips.itertuples():
                if row.end_station_id in station_id_to_idx:
                    destination_probabilities[station_id_to_idx[row.end_station_id]] = row.count
            if destination_probabilities.sum() == 0:
                continue
            dest_station_idx = int(self.rng.choice(self.num_stations, p=destination_probabilities / destination_probabilities.sum()))

            # Check system constraints: bike available at origin and open slot at destination
            if self.station_inventories[origin_station_idx] >= 1 and self.station_inventories[dest_station_idx] < self.cap[dest_station_idx]:
                self.station_inventories[origin_station_idx] -= 1
                self.station_inventories[dest_station_idx] += 1
                completed_trips += 1
                step_reward += 1.0
            else:
                unmet_requests += 1
                step_reward -= 2.0
                
        self.current_time_hours += remaining_minutes / 60
        return step_reward, completed_trips, unmet_requests
    
    def reset(self,seed=None,options=None):
        """
        Clear existing episode trajectories and reset to initial states for all stations.
        Currently sets all stations to be between 30% and 70% filled, picks a random starting 
        weekday and places the truck at the first station.
        """
        super().reset(seed=seed)
        if seed is not None: 
            self.rng=np.random.default_rng(seed)
        frac=float(self.rng.uniform(.3,.7))
        self.station_inventories =np.floor(self.cap*frac).astype(int)
        self.current_truck_load =self.truck_capacity//2
        self.truck_location_idx =0
        self.current_time_hours =0.0
        self.start_weekday =int(self.rng.integers(0,7))
        self.total_distance_driven =0.
        self.total_travel_minutes =0.
        self.total_completed_trips =0
        self.total_unmet_requests =0
        self.station_imbalance_minutes =np.zeros(self.num_stations)
        return self._obs(),{"station_ids":self.station_ids}
    def step(self,action):
        """
        Here the truck travels for T mins to the next station from a to b and this time (T) is then
        used in customer_events to calculate what happened while the truck was moving. Then the truck
        takes the decision_minutes to simulate loading and unloading bikes, which is again used in customer_events
        and then it goes off to the new station. It recieves rewards and penalties based on customer_events
        and a small movement cost to promote efficiency on distance travelled. 
        """
        target_station_idx, transfer_level_idx = map(int, action)
        requested_transfer = int(self.transfer_options[transfer_level_idx])
        travel_distance_km = self._dist(self.truck_location_idx, target_station_idx)
        travel_minutes = (travel_distance_km / self.truck_speed_kmh) * 60
        
        # Process customer demand occurring while the truck is in transit
        reward, completed, unmet = self._customer_events(travel_minutes)
        self.total_distance_driven += travel_distance_km
        self.total_travel_minutes += travel_minutes
        self.truck_location_idx = target_station_idx
        
        actual_transfer_amount = 0
        if requested_transfer > 0:  # Unload bikes from truck to station
            actual_transfer_amount = min([
                requested_transfer, 
                self.current_truck_load, 
                int(self.cap[target_station_idx] - self.station_inventories[target_station_idx])])
            self.station_inventories[target_station_idx] += actual_transfer_amount
            self.current_truck_load -= actual_transfer_amount
        elif requested_transfer < 0:  # Load bikes from station to truck
            actual_transfer_amount = min([
                -requested_transfer, 
                int(self.station_inventories[target_station_idx]), 
                self.truck_capacity - self.current_truck_load])
            self.station_inventories[target_station_idx] -= actual_transfer_amount
            self.current_truck_load += actual_transfer_amount
            
        # Process customer events during the truck's operational buffer time at the station
        buffer_reward, buffer_completed, buffer_unmet = self._customer_events(self.decision_minutes)
        reward += buffer_reward
        completed += buffer_completed
        unmet += buffer_unmet
        
        # Calculate step penalties
        reward -= (0.025 * travel_distance_km) + (0.0005 * travel_minutes)
        reward -= 0.01 * abs(requested_transfer - actual_transfer_amount)
        self.total_completed_trips += completed
        self.total_unmet_requests += unmet
        # Track the time stations spend completely empty or completely full
        self.station_imbalance_minutes += (self.decision_minutes / 60) * (
            (self.station_inventories == 0) | (self.station_inventories >= self.cap)
        )
        is_terminated = self.current_time_hours >= self.episode_hours
        step_info = {
            "completed_trips": completed,
            "unmet_requests": unmet,
            "distance_km": np.round(travel_distance_km,2),
            "truck_load": self.current_truck_load,
            "transfer": actual_transfer_amount
        }
        
        if is_terminated:
            step_info.update({
                "episode_completed": self.total_completed_trips,
                "episode_unmet": self.total_unmet_requests,
                "distance_total_km": self.total_distance_driven,
                "empty_full_fraction": float(self.station_imbalance_minutes.sum() / (self.num_stations * self.episode_hours))
            })
            
        return self._obs(), float(reward), is_terminated, False, step_info
    
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    STATIONS_PATH = BASE_DIR / "data" / "processed" / "stations.csv"
    DEMAND_PATH = BASE_DIR / "data" / "processed" / "od_demand.parquet"
    stations_df = pd.read_csv(STATIONS_PATH)
    od_df = pd.read_parquet(DEMAND_PATH)
    env = BikeShareEnv(stations=stations_df, od=od_df)
    obs, info = env.reset(seed=1508)
    print("Test Run")
    terminated = False
    step_count = 0
    total_reward = 0.0
    
    while not terminated:
        action = env.action_space.sample()
        current_timestamp = env.get_formatted_time()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        step_count += 1
        station_id = env.station_ids[action[0]]
        
        if step_count <= 10:
            print(f"[{current_timestamp}] Sample Action -> Station Index Picked: {action[0]} (Station ID: {station_id}), Transfer Level Code: {action[1]}")
            print(f"Step Info Dictionary Payload -> {step_info}, reward: {reward}")
        if np.isnan(next_obs).any() or np.isinf(next_obs).any():
            print(f"error at step {step_count}!")
