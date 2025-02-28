# Import necessary modules from FastAPI and related libraries.
from fastapi import FastAPI, HTTPException, Query  # FastAPI framework and HTTP exceptions, query parameters support
from fastapi.middleware.cors import CORSMiddleware  # For handling Cross-Origin Resource Sharing
from pydantic import BaseModel  # For data validation and serialization
from typing import List  # To define list types in type hints
import threading, time, random  # Modules for threading (simultaneous execution), time delays, and randomness
from datetime import datetime  # For working with dates and times
from collections import defaultdict  # For creating dictionaries with a default value

# Create an instance of the FastAPI application.
app = FastAPI()

# Add middleware to allow cross-origin requests (from any origin).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],         # Allow all HTTP methods
    allow_headers=["*"],         # Allow all headers
)

# Define a helper function to get the current timestamp in [HH:MM:SS] format.
def get_timestamp():
    return "[" + datetime.now().strftime('%H:%M:%S') + "]"

# ------------------------------
# Tank Class: Represents a single water tank.
# ------------------------------
class Tank:
    def __init__(self, capacity: float):
        self.capacity = capacity              # Maximum capacity of the tank
        self.water_level = capacity           # Start with a full tank
        self.consumption_rate = 0.005 * capacity  # Water consumption rate (0.5% of capacity per second)
        self.pump_rate = 0.03 * capacity       # Water refill rate (3% of capacity per second)
        self.low_threshold = 0.3 * capacity    # Threshold (30% of capacity) below which water is considered low
        self.state = "idle"                   # Initial state is "idle" (not active or refilling)
        self.last_event = get_timestamp() + " Initialized."  # Log the initialization event

# ------------------------------
# MultiTankSystem Class: Manages multiple tanks and the simulation.
# ------------------------------
class MultiTankSystem:
    def __init__(self, tanks: List[Tank], simulation_speed: float = 1.0):
        self.tanks = tanks                              # List of tank instances
        self.simulation_speed = simulation_speed        # Simulation speed (seconds per update cycle)
        self.manual_override = False                    # Flag to enable/disable manual control
        self.running = True                             # Flag indicating if the simulation is running
        self.lock = threading.Lock()                    # Lock for thread-safe operations
        self.active_event = threading.Event()           # Event to control simulation cycles
        self.active_event.set()                         # Activate simulation cycle by default
        self.alerts = []                                # List to log alert messages
        self.operational_cost = 0.0                     # Accumulated cost of operation over time
        self.consumption_log = []                       # Log of water consumed per cycle (timestamp, consumption)

        # Set the first tank to "active" and the rest to "idle".
        if self.tanks:
            self.tanks[0].state = "active"
            self.tanks[0].last_event = get_timestamp() + " Set as active."
            # Loop through remaining tanks and set their state to idle.
            for tank in self.tanks[1:]:
                tank.state = "idle"

    def rebalance(self):
        # Rebalanced water among tanks for better efficiency.
        # If one tank is over 90% full and another is below 40% full, transfer water.
        for i in range(len(self.tanks)):
            for j in range(len(self.tanks)):
                if i == j:
                    continue  # Skip if comparing the same tank
                tank_i = self.tanks[i]
                tank_j = self.tanks[j]
                if (tank_i.water_level > 0.9 * tank_i.capacity and
                    tank_j.water_level < 0.4 * tank_j.capacity):
                    # Calculate the maximum possible transfer amount.
                    transfer_amount = min(tank_i.water_level - 0.9 * tank_i.capacity,
                                          0.4 * tank_j.capacity - tank_j.water_level)
                    if transfer_amount > 0:
                        # Deduct water from the fuller tank and add to the emptier tank.
                        tank_i.water_level -= transfer_amount
                        tank_j.water_level += transfer_amount
                        # Create an alert message and update both tanks' last event.
                        msg = get_timestamp() + f" Rebalanced: transferred {transfer_amount:.2f} from tank {i} to tank {j}."
                        self.alerts.append(msg)
                        tank_i.last_event = msg
                        tank_j.last_event = msg

    def update(self):
        # Main update method to be called periodically (each simulation cycle).
        with self.lock:
            cycle_consumption = 0.0  # Initialize water consumed during this cycle

            # Iterate through each tank to update water levels.
            for tank in self.tanks:
                if tank.state == "active":
                    # Calculate water consumption with a random factor for variability.
                    random_factor = random.uniform(0.8, 1.2)
                    consumption = tank.consumption_rate * random_factor
                    tank.water_level -= consumption  # Subtract water consumed
                    cycle_consumption += consumption  # Accumulate consumption for logging
                    if tank.water_level < 0:
                        tank.water_level = 0  # Ensure water level doesn't drop below zero
                        self.alerts.append(get_timestamp() + f" Tank {self.tanks.index(tank)} is empty.")
                elif tank.state == "refill":
                    # Increase water level when refilling.
                    tank.water_level += tank.pump_rate
                    if tank.water_level > tank.capacity:
                        tank.water_level = tank.capacity  # Prevent overfilling

            # Log the water consumption if any was recorded during this cycle.
            if cycle_consumption > 0:
                self.consumption_log.append((datetime.now(), cycle_consumption))

            # AUTOMATIC MODE: Only execute this section if manual override is off.
            if not self.manual_override:
                active_tank = None
                # Find the first tank in the "active" state.
                for tank in self.tanks:
                    if tank.state == "active":
                        active_tank = tank
                        break

                # If an active tank exists and its water is low, switch it to refill.
                if active_tank and active_tank.water_level <= active_tank.low_threshold:
                    active_tank.state = "refill"
                    msg = get_timestamp() + " Low water; switching to refill."
                    active_tank.last_event = msg
                    self.alerts.append(f"Tank {self.tanks.index(active_tank)}: " + msg)
                    current_index = self.tanks.index(active_tank)
                    # Loop through tanks to activate the next idle one.
                    for i in range(1, len(self.tanks)):
                        next_index = (current_index + i) % len(self.tanks)
                        if self.tanks[next_index].state == "idle":
                            self.tanks[next_index].state = "active"
                            activation_msg = get_timestamp() + " Activated for discharge."
                            self.tanks[next_index].last_event = activation_msg
                            self.alerts.append(f"Tank {next_index}: " + activation_msg)
                            break

                # If there is no active tank, activate the first tank in idle state.
                if not any(tank.state == "active" for tank in self.tanks):
                    for tank in self.tanks:
                        if tank.state == "idle":
                            tank.state = "active"
                            activation_msg = get_timestamp() + " Activated (none active)."
                            tank.last_event = activation_msg
                            self.alerts.append(f"Tank {self.tanks.index(tank)}: " + activation_msg)
                            break

                # For tanks refilling: if the tank is full, set it to idle.
                for tank in self.tanks:
                    if tank.state == "refill" and tank.water_level >= tank.capacity:
                        tank.state = "idle"
                        msg = get_timestamp() + " Refilled; now idle."
                        tank.last_event = msg
                        self.alerts.append(f"Tank {self.tanks.index(tank)}: " + msg)

                # Call rebalance to even out water levels among tanks.
                self.rebalance()

            # MANUAL OVERRIDE MODE: Check if the active tank is critically low.
            else:
                active_tank = None
                for tank in self.tanks:
                    if tank.state == "active":
                        active_tank = tank
                        break
                if active_tank:
                    # Define critical threshold as low threshold minus 5% of capacity.
                    critical_threshold = active_tank.low_threshold - 0.05 * active_tank.capacity
                    if active_tank.water_level < critical_threshold:
                        # If critically low, exit manual mode and log the event.
                        self.manual_override = False
                        msg = get_timestamp() + " Critically low; auto mode restored."
                        active_tank.last_event = msg
                        self.alerts.append(f"Tank {self.tanks.index(active_tank)}: " + msg)

            # Update operational cost based on current tank states.
            for tank in self.tanks:
                if tank.state == "refill":
                    self.operational_cost += 0.1 * self.simulation_speed
                elif tank.state == "active":
                    self.operational_cost += 0.01 * self.simulation_speed

    def get_sensor_reading(self, tank: Tank) -> float:
        # Return the current water level plus a small random noise.
        noise = random.uniform(-0.5, 0.5)
        return tank.water_level + noise

# ------------------------------
# Helper Function: Aggregate consumption data.
# ------------------------------
def aggregate_consumption(consumption_log, period: str):
    # Create a dictionary that defaults to float (0.0) for each key.
    aggregated = defaultdict(float)
    # Loop through each (timestamp, consumption) tuple in the log.
    for ts, cons in consumption_log:
        # Choose a key format based on the period.
        if period == "day":
            key = ts.strftime("%H:00")  # Group by hour
        elif period == "month":
            key = ts.strftime("%Y-%m-%d")  # Group by day
        elif period == "year":
            key = ts.strftime("%Y-%m")  # Group by month
        else:
            key = ts.strftime("%Y-%m-%d %H:%M:%S")
        aggregated[key] += cons  # Sum consumption for the key

    # Sort the keys appropriately.
    if period == "day":
        sorted_keys = sorted(aggregated.keys(), key=lambda x: int(x.split(":")[0]))
    else:
        sorted_keys = sorted(aggregated.keys())

    # Return the aggregated data as a list of dictionaries.
    return [{"time": key, "consumption": aggregated[key]} for key in sorted_keys]

# ------------------------------
# Pydantic Models for API Responses
# ------------------------------

# Model for tank data returned by the API.
class TankData(BaseModel):
    id: int
    capacity: float
    water_level: float
    state: str
    last_event: str
    sensor: float

# Model for system data (e.g., manual override, simulation state).
class SystemData(BaseModel):
    manual_override: bool
    deactivated: bool  # True if the simulation cycle is paused

# Model for overall statistics.
class StatsData(BaseModel):
    total_capacity: float
    total_water: float
    overall_fullness: float
    active_count: int
    refill_count: int
    idle_count: int
    operational_cost: float

# Model for actions performed on a tank (e.g., change state).
class Action(BaseModel):
    action: str  # Allowed values: "active", "refill", or "idle"

# Model for initializing the system with a list of tank capacities.
class InitializationData(BaseModel):
    capacities: List[float]

# Model for consumption data response.
class ConsumptionDataModel(BaseModel):
    time: str
    consumption: float

# ------------------------------
# API Endpoints
# ------------------------------

# Endpoint to initialize the system with specified tank capacities.
@app.post("/api/initialize", response_model=List[TankData])
def initialize_system_endpoint(data: InitializationData):
    global system
    system.running = False  # Stop the current simulation
    new_tanks = [Tank(cap) for cap in data.capacities]  # Create tanks from provided capacities
    system = MultiTankSystem(new_tanks, simulation_speed=1.0)  # Create a new system
    start_simulation()  # Start the simulation thread
    return get_tanks()  # Return the current tank data

# Endpoint to get data for all tanks.
@app.get("/api/tanks", response_model=List[TankData])
def get_tanks():
    with system.lock:
        tanks_list = []
        # Loop through tanks and create a list of TankData models.
        for idx, tank in enumerate(system.tanks):
            tanks_list.append(TankData(
                id=idx,
                capacity=tank.capacity,
                water_level=tank.water_level,
                state=tank.state,
                last_event=tank.last_event,
                sensor=system.get_sensor_reading(tank)
            ))
        return tanks_list

# Endpoint to get data for a specific tank by its ID.
@app.get("/api/tanks/{tank_id}", response_model=TankData)
def get_tank(tank_id: int):
    with system.lock:
        if tank_id < 0 or tank_id >= len(system.tanks):
            raise HTTPException(status_code=404, detail="Tank not found")
        tank = system.tanks[tank_id]
        return TankData(
            id=tank_id,
            capacity=tank.capacity,
            water_level=tank.water_level,
            state=tank.state,
            last_event=tank.last_event,
            sensor=system.get_sensor_reading(tank)
        )

# Endpoint to set the state of a specific tank.
@app.post("/api/tanks/{tank_id}/set_state", response_model=TankData)
def set_tank_state(tank_id: int, action: Action):
    with system.lock:
        if tank_id < 0 or tank_id >= len(system.tanks):
            raise HTTPException(status_code=404, detail="Tank not found")
        if action.action not in ("active", "refill", "idle"):
            raise HTTPException(status_code=400, detail="Invalid action")
        tank = system.tanks[tank_id]
        tank.state = action.action  # Update the tank's state
        tank.last_event = get_timestamp() + " Manually set to " + action.action + "."
        system.alerts.append(f"Tank {tank_id}: " + tank.last_event)
        return TankData(
            id=tank_id,
            capacity=tank.capacity,
            water_level=tank.water_level,
            state=tank.state,
            last_event=tank.last_event,
            sensor=system.get_sensor_reading(tank)
        )

# Endpoint to get overall system information.
@app.get("/api/system", response_model=SystemData)
def get_system_info():
    with system.lock:
        return SystemData(
            manual_override=system.manual_override,
            deactivated=(not system.active_event.is_set())
        )

# Endpoint to get system statistics.
@app.get("/api/statistics", response_model=StatsData)
def get_statistics():
    with system.lock:
        total_capacity = sum(tank.capacity for tank in system.tanks)
        total_water = sum(tank.water_level for tank in system.tanks)
        active_count = sum(1 for tank in system.tanks if tank.state == "active")
        refill_count = sum(1 for tank in system.tanks if tank.state == "refill")
        idle_count = sum(1 for tank in system.tanks if tank.state == "idle")
        overall_fullness = (total_water / total_capacity * 100) if total_capacity > 0 else 0
    return StatsData(
        total_capacity=total_capacity,
        total_water=total_water,
        overall_fullness=overall_fullness,
        active_count=active_count,
        refill_count=refill_count,
        idle_count=idle_count,
        operational_cost=system.operational_cost
    )

# Endpoint to toggle manual override mode.
@app.post("/api/system/toggle_manual", response_model=SystemData)
def toggle_manual():
    with system.lock:
        system.manual_override = not system.manual_override  # Flip manual override state
        return SystemData(
            manual_override=system.manual_override,
            deactivated=(not system.active_event.is_set())
        )

# Endpoint to deactivate the simulation cycle.
@app.post("/api/system/deactivate_cycle", response_model=SystemData)
def deactivate_cycle():
    with system.lock:
        system.active_event.clear()  # Pause the simulation cycle
        return SystemData(
            manual_override=system.manual_override,
            deactivated=(not system.active_event.is_set())
        )

# Endpoint to activate the simulation cycle.
@app.post("/api/system/activate_cycle", response_model=SystemData)
def activate_cycle():
    with system.lock:
        system.active_event.set()  # Resume the simulation cycle
        return SystemData(
            manual_override=system.manual_override,
            deactivated=(not system.active_event.is_set())
        )

# Endpoint to retrieve all alert messages.
@app.get("/api/alerts", response_model=List[str])
def get_alerts():
    with system.lock:
        return system.alerts

# Endpoint to clear all alerts.
@app.post("/api/alerts/clear")
def clear_alerts():
    with system.lock:
        system.alerts = []  # Clear the alert list
        return {"message": "Alerts cleared"}

# Endpoint to get consumption data aggregated by period.
@app.get("/api/consumption", response_model=List[ConsumptionDataModel])
def get_consumption_data(period: str = Query("day", enum=["day", "month", "year"])):
    with system.lock:
        data = aggregate_consumption(system.consumption_log, period)
    return data

# ------------------------------
# Simulation Loop: Runs in a separate thread.
# ------------------------------

# Global variable for the simulation thread.
simulation_thread = None

# Function that runs the simulation loop.
def simulation_loop():
    # Run continuously while the simulation is set to run.
    while system.running:
        system.active_event.wait()  # Wait until simulation cycle is active
        system.update()             # Update the system (water levels, state changes, etc.)
        time.sleep(system.simulation_speed)  # Wait for the defined simulation speed duration

# Function to start the simulation in a separate thread.
def start_simulation():
    global simulation_thread
    # Only start the simulation thread if it is not already running.
    if simulation_thread is None or not simulation_thread.is_alive():
        simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
        simulation_thread.start()

# ------------------------------
# Create and start the simulation.
# ------------------------------

# Create a global MultiTankSystem with three tanks and start the simulation.
system = MultiTankSystem([Tank(100), Tank(150), Tank(120)], simulation_speed=1.0)
start_simulation()
