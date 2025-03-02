# Import FastAPI framework classes and functions for creating API endpoints and handling errors.
from fastapi import FastAPI, HTTPException, \
    Query  # FastAPI creates the API; HTTPException handles errors; Query validates query parameters.
# Import CORS middleware to allow cross-origin requests.
from fastapi.middleware.cors import CORSMiddleware  # Allows the API to be accessed from different origins.
# Import StreamingResponse to stream binary data (used later to send the chart image).
from fastapi.responses import StreamingResponse  # Used for returning the generated image as a response.
# Import BaseModel from pydantic to define data models for request/response bodies.
from pydantic import BaseModel  # Provides model validation and serialization.
# Import List from typing for type annotations.
from typing import List  # Indicates that a variable will hold a list of items.
# Import threading, time, and random for concurrent execution, delays, and randomness.
import threading, time, random  # threading allows concurrent execution; time provides delays; random adds variability.
# Import datetime for working with date and time.
from datetime import datetime  # Used for timestamps and logging.
# Import defaultdict from collections to aggregate data with default values.
from collections import defaultdict  # Automatically initializes dictionary keys with a default type.
# Import matplotlib for creating graphs.
import matplotlib.pyplot as plt  # Used to generate the line graph.
# Import BytesIO to handle in-memory binary streams (for image data).
from io import BytesIO  # Used to save the plot image in memory.

# Create an instance of the FastAPI application.
app = FastAPI()  # This instance will handle all incoming API requests.

# Add CORS middleware to the app to allow requests from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any domain.
    allow_credentials=True,  # Allow credentials (cookies, authorization headers, etc.).
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.).
    allow_headers=["*"],  # Allow all HTTP headers.
)

# Global variable for the simulation thread (used to run the simulation loop concurrently).
simulation_thread = None  # Will hold the thread that runs the simulation loop.


# Define a helper function to generate a timestamp string in the format [HH:MM:SS].
def get_timestamp():
    return "[" + datetime.now().strftime('%H:%M:%S') + "]"  # Returns the current time wrapped in square brackets.


# Define the Tank class which represents a single water tank.
class Tank:
    def __init__(self, capacity: float):
        self.capacity = capacity  # Set the maximum capacity of the tank.
        self.water_level = capacity  # Initialize the water level to full capacity.
        self.consumption_rate = 0.005 * capacity  # Set consumption rate to 0.5% of capacity per cycle.
        self.pump_rate = 0.03 * capacity  # Set pump (refill) rate to 3% of capacity per cycle.
        self.low_threshold = 0.3 * capacity  # Define low threshold at 30% of capacity.
        self.state = "idle"  # Initial state is "idle" (not actively discharging).
        self.last_event = get_timestamp() + " Initialized."  # Log the initialization event with a timestamp.


# Define the MultiTankSystem class to manage multiple tanks and run the simulation.
class MultiTankSystem:
    def __init__(self, tanks: List[Tank], simulation_speed: float = 1.0):
        self.tanks = tanks  # Store the list of Tank objects.
        self.simulation_speed = simulation_speed  # Store the simulation speed (time delay between cycles).
        self.manual_override = False  # Flag to indicate if manual override is enabled.
        self.running = True  # Control flag for running the simulation loop.
        self.lock = threading.Lock()  # Lock to ensure thread-safe modifications.
        self.active_event = threading.Event()  # Event to pause/resume the simulation cycle.
        self.active_event.set()  # Set the event so that the simulation runs.
        self.alerts = []  # List to store alert messages and logs.
        self.operational_cost = 0.0  # Variable to track operational cost over simulation cycles.
        self.consumption_log = []  # Log of water consumption per cycle.

        # If there is at least one tank, set the first tank as "active" (discharging water).
        if self.tanks:
            self.tanks[0].state = "active"  # Mark the first tank as active.
            self.tanks[0].last_event = get_timestamp() + " Set as active."  # Log the state change.
            # Ensure all remaining tanks are in "idle" state.
            for tank in self.tanks[1:]:
                tank.state = "idle"

    # Function to balance water levels between tanks.
    def balance(self):
        # Loop over every pair of tanks.
        for i in range(len(self.tanks)):
            for j in range(len(self.tanks)):
                if i == j:
                    continue  # Skip comparing the same tank.
                tank_i = self.tanks[i]  # Tank to potentially transfer water from.
                tank_j = self.tanks[j]  # Tank to potentially receive water.
                # Check if tank_i is more than 90% full and tank_j is less than 40% full.
                if (tank_i.water_level > 0.9 * tank_i.capacity and
                        tank_j.water_level < 0.4 * tank_j.capacity):
                    # Calculate the transfer amount without over-draining or overfilling.
                    transfer_amount = min(tank_i.water_level - 0.9 * tank_i.capacity,
                                          0.4 * tank_j.capacity - tank_j.water_level)
                    if transfer_amount > 0:
                        # Transfer water between tanks.
                        tank_i.water_level -= transfer_amount
                        tank_j.water_level += transfer_amount
                        # Create a log message with the details of the transfer.
                        msg = get_timestamp() + f" Rebalanced: transferred {transfer_amount:.2f} from tank {i} to tank {j}."
                        self.alerts.append(msg)  # Add the alert message to the list.
                        tank_i.last_event = msg  # Update the last event for the sending tank.
                        tank_j.last_event = msg  # Update the last event for the receiving tank.

    # Function that performs a single simulation update cycle.
    def update(self):
        with self.lock:  # Ensure that the update is thread-safe.
            cycle_consumption = 0.0  # Initialize consumption for this cycle.
            # Iterate over all tanks to update their water levels.
            for tank in self.tanks:
                if tank.state == "active":  # If the tank is actively discharging water.
                    random_factor = random.uniform(0.8, 1.2)  # Add randomness to the consumption.
                    consumption = tank.consumption_rate * random_factor  # Calculate water consumed.
                    tank.water_level -= consumption  # Deduct the consumed water.
                    cycle_consumption += consumption  # Accumulate the cycle's total consumption.
                    # If water level drops below zero, set it to zero and log an alert.
                    if tank.water_level < 0:
                        tank.water_level = 0
                        self.alerts.append(get_timestamp() + f" Tank {self.tanks.index(tank)} is empty.")
                elif tank.state == "refill":  # If the tank is in refill mode.
                    tank.water_level += tank.pump_rate  # Increase water level by pump rate.
                    if tank.water_level > tank.capacity:
                        tank.water_level = tank.capacity  # Ensure water level does not exceed capacity.

            # If any consumption occurred during this cycle, add it to the consumption log.
            if cycle_consumption > 0:
                self.consumption_log.append((datetime.now(), cycle_consumption))

            # AUTOMATIC MODE: If manual override is not active, adjust tank states automatically.
            if not self.manual_override:
                active_tank = None  # Initialize variable to hold the current active tank.
                # Find the first tank in active state.
                for tank in self.tanks:
                    if tank.state == "active":
                        active_tank = tank
                        break

                # If an active tank exists and its water level is low, change its state to refill.
                if active_tank and active_tank.water_level <= active_tank.low_threshold:
                    active_tank.state = "refill"  # Switch to refill mode.
                    msg = get_timestamp() + " Low water; switching to refill."  # Create a log message.
                    active_tank.last_event = msg  # Update the tank's last event.
                    self.alerts.append(f"Tank {self.tanks.index(active_tank)}: " + msg)  # Log the alert.
                    current_index = self.tanks.index(active_tank)  # Get the index of the active tank.
                    # Activate the next idle tank using modular arithmetic.
                    for i in range(1, len(self.tanks)):
                        next_index = (current_index + i) % len(self.tanks)
                        if self.tanks[next_index].state == "idle":
                            self.tanks[next_index].state = "active"
                            activation_msg = get_timestamp() + " Activated for discharge."  # Log activation.
                            self.tanks[next_index].last_event = activation_msg
                            self.alerts.append(f"Tank {next_index}: " + activation_msg)
                            break

                # If no tank is currently active, activate the first idle tank.
                if not any(tank.state == "active" for tank in self.tanks):
                    for tank in self.tanks:
                        if tank.state == "idle":
                            tank.state = "active"
                            activation_msg = get_timestamp() + " Activated (none active)."
                            tank.last_event = activation_msg
                            self.alerts.append(f"Tank {self.tanks.index(tank)}: " + activation_msg)
                            break

                # For any tank in refill mode that becomes full, switch its state to idle.
                for tank in self.tanks:
                    if tank.state == "refill" and tank.water_level >= tank.capacity:
                        tank.state = "idle"
                        msg = get_timestamp() + " Refilled; now idle."
                        tank.last_event = msg
                        self.alerts.append(f"Tank {self.tanks.index(tank)}: " + msg)

                # Call balance to redistribute water among tanks if needed.
                self.balance()

            # MANUAL OVERRIDE MODE: If manual override is enabled, check if active tank needs to revert to auto mode.
            else:
                active_tank = None
                for tank in self.tanks:
                    if tank.state == "active":
                        active_tank = tank
                        break
                if active_tank:
                    # Define a critical threshold as 5% lower than the normal low threshold.
                    critical_threshold = active_tank.low_threshold - 0.05 * active_tank.capacity
                    # If water level falls below the critical threshold, disable manual override.
                    if active_tank.water_level < critical_threshold:
                        self.manual_override = False
                        msg = get_timestamp() + " Critically low; auto mode restored."
                        active_tank.last_event = msg
                        self.alerts.append(f"Tank {self.tanks.index(active_tank)}: " + msg)

            # Update operational cost: Refilling is costlier (0.1 per cycle) than active operation (0.01 per cycle).
            for tank in self.tanks:
                if tank.state == "refill":
                    self.operational_cost += 0.1 * self.simulation_speed
                elif tank.state == "active":
                    self.operational_cost += 0.01 * self.simulation_speed

    # Function to simulate a sensor reading by adding noise to the actual water level.
    def get_sensor_reading(self, tank: Tank) -> float:
        noise = random.uniform(-0.5, 0.5)  # Generate a small random noise.
        return tank.water_level + noise  # Return the water level with noise.


# Helper function to aggregate water consumption data over a given period.
def aggregate_consumption(consumption_log, period: str):
    aggregated = defaultdict(float)  # Create a dictionary with default float values.
    # Iterate over the log entries, which are tuples of (timestamp, consumption).
    for ts, cons in consumption_log:
        if period == "day":
            key = ts.strftime("%H:00")  # Group by hour if period is "day".
        elif period == "month":
            key = ts.strftime("%Y-%m-%d")  # Group by day if period is "month".
        elif period == "year":
            key = ts.strftime("%Y-%m")  # Group by month if period is "year".
        else:
            key = ts.strftime("%Y-%m-%d %H:%M:%S")  # Use full timestamp for any other case.
        aggregated[key] += cons  # Sum the consumption for each key.

    # Sort the keys in chronological order.
    if period == "day":
        sorted_keys = sorted(aggregated.keys(), key=lambda x: int(x.split(":")[0]))
    else:
        sorted_keys = sorted(aggregated.keys())

    # Return the aggregated data as a list of dictionaries.
    return [{"time": key, "consumption": aggregated[key]} for key in sorted_keys]


# Pydantic model for returning tank data in API responses.
class TankData(BaseModel):
    id: int  # Unique identifier for the tank.
    capacity: float  # Maximum capacity of the tank.
    water_level: float  # Current water level.
    state: str  # Operational state ("active", "refill", "idle").
    last_event: str  # Last event message logged for this tank.
    sensor: float  # Sensor reading (with noise).


# Pydantic model for returning overall system status.
class SystemData(BaseModel):
    manual_override: bool  # Whether manual override is enabled.
    deactivated: bool  # Whether the simulation cycle is paused.


# Pydantic model for returning simulation statistics.
class StatsData(BaseModel):
    total_capacity: float  # Sum of all tank capacities.
    total_water: float  # Total water remaining in all tanks.
    overall_fullness: float  # Percentage fullness across all tanks.
    active_count: int  # Number of tanks in active state.
    refill_count: int  # Number of tanks in refill state.
    idle_count: int  # Number of tanks in idle state.
    operational_cost: float  # Total operational cost accumulated.


# Pydantic model for actions to change a tank's state.
class Action(BaseModel):
    action: str  # The desired action ("active", "refill", or "idle").


# Pydantic model for initializing the system with tank capacities.
class InitializationData(BaseModel):
    capacities: List[float]  # List of capacities for each tank.


# Pydantic model for consumption data returned by the API.
class ConsumptionDataModel(BaseModel):
    time: str  # Time label for the consumption data.
    consumption: float  # Total consumption recorded during that time period.


# Define the simulation loop that continuously updates the system.
def simulation_loop():
    # Run the loop while the simulation is set to run.
    while system.running:
        system.active_event.wait()  # Wait until the simulation cycle is activated.
        system.update()  # Run an update cycle to simulate changes.
        time.sleep(system.simulation_speed)  # Delay for the specified simulation speed.


# Function to start the simulation thread if it's not already running.
def start_simulation():
    global simulation_thread  # Access the global simulation_thread variable.
    # Check if the simulation thread does not exist or is not active.
    if simulation_thread is None or not simulation_thread.is_alive():
        # Create and start a new thread to run the simulation_loop.
        simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
        simulation_thread.start()


# API endpoint to initialize the simulation with a new set of tanks.
@app.post("/api/initialize", response_model=List[TankData])
def initialize_system_endpoint(data: InitializationData):
    global system  # Access the global system variable.
    system.running = False  # Stop the current simulation loop.
    new_tanks = [Tank(cap) for cap in data.capacities]  # Create a new Tank for each capacity provided.
    system = MultiTankSystem(new_tanks, simulation_speed=1.0)  # Create a new MultiTankSystem.
    start_simulation()  # Start the simulation thread.
    return get_tanks()  # Return the current state of all tanks.


# API endpoint to retrieve the list of all tanks and their data.
@app.get("/api/tanks", response_model=List[TankData])
def get_tanks():
    with system.lock:  # Ensure thread-safe access to the system data.
        tanks_list = []  # Initialize an empty list to hold tank data.
        for idx, tank in enumerate(system.tanks):  # Iterate over each tank.
            tanks_list.append(TankData(
                id=idx,
                capacity=tank.capacity,
                water_level=tank.water_level,
                state=tank.state,
                last_event=tank.last_event,
                sensor=system.get_sensor_reading(tank)  # Get sensor reading with noise.
            ))
        return tanks_list  # Return the list of tank data.


# API endpoint to retrieve data for a specific tank identified by tank_id.
@app.get("/api/tanks/{tank_id}", response_model=TankData)
def get_tank(tank_id: int):
    with system.lock:  # Ensure thread-safe access.
        # Check if the tank_id is valid.
        if tank_id < 0 or tank_id >= len(system.tanks):
            raise HTTPException(status_code=404, detail="Tank not found")
        tank = system.tanks[tank_id]  # Retrieve the specified tank.
        return TankData(
            id=tank_id,
            capacity=tank.capacity,
            water_level=tank.water_level,
            state=tank.state,
            last_event=tank.last_event,
            sensor=system.get_sensor_reading(tank)
        )


# API endpoint to manually set the state of a tank.
@app.post("/api/tanks/{tank_id}/set_state", response_model=TankData)
def set_tank_state(tank_id: int, action: Action):
    with system.lock:  # Ensure thread-safe access.
        # Validate the tank_id.
        if tank_id < 0 or tank_id >= len(system.tanks):
            raise HTTPException(status_code=404, detail="Tank not found")
        # Validate that the action is one of the allowed values.
        if action.action not in ("active", "refill", "idle"):
            raise HTTPException(status_code=400, detail="Invalid action")
        tank = system.tanks[tank_id]  # Retrieve the specified tank.
        tank.state = action.action  # Set the tank's state.
        tank.last_event = get_timestamp() + " Manually set to " + action.action + "."  # Log the manual change.
        system.alerts.append(f"Tank {tank_id}: " + tank.last_event)  # Add the alert.
        return TankData(
            id=tank_id,
            capacity=tank.capacity,
            water_level=tank.water_level,
            state=tank.state,
            last_event=tank.last_event,
            sensor=system.get_sensor_reading(tank)
        )


# API endpoint to get overall system information.
@app.get("/api/system", response_model=SystemData)
def get_system_info():
    with system.lock:
        return SystemData(
            manual_override=system.manual_override,
            deactivated=(not system.active_event.is_set())  # True if simulation cycle is paused.
        )


# API endpoint to retrieve overall simulation statistics.
@app.get("/api/statistics", response_model=StatsData)
def get_statistics():
    with system.lock:
        total_capacity = sum(tank.capacity for tank in system.tanks)  # Sum of all tank capacities.
        total_water = sum(tank.water_level for tank in system.tanks)  # Sum of current water levels.
        active_count = sum(1 for tank in system.tanks if tank.state == "active")  # Count active tanks.
        refill_count = sum(1 for tank in system.tanks if tank.state == "refill")  # Count tanks refilling.
        idle_count = sum(1 for tank in system.tanks if tank.state == "idle")  # Count idle tanks.
        overall_fullness = (
                    total_water / total_capacity * 100) if total_capacity > 0 else 0  # Calculate overall fullness percentage.
    return StatsData(
        total_capacity=total_capacity,
        total_water=total_water,
        overall_fullness=overall_fullness,
        active_count=active_count,
        refill_count=refill_count,
        idle_count=idle_count,
        operational_cost=system.operational_cost  # Return the operational cost.
    )


# API endpoint to toggle manual override mode.
@app.post("/api/system/toggle_manual", response_model=SystemData)
def toggle_manual():
    with system.lock:
        system.manual_override = not system.manual_override  # Flip the manual override flag.
        return SystemData(
            manual_override=system.manual_override,
            deactivated=(not system.active_event.is_set())
        )


# API endpoint to pause the simulation cycle.
@app.post("/api/system/deactivate_cycle", response_model=SystemData)
def deactivate_cycle():
    with system.lock:
        system.active_event.clear()  # Clear the event to pause the simulation.
        return SystemData(
            manual_override=system.manual_override,
            deactivated=(not system.active_event.is_set())
        )


# API endpoint to resume the simulation cycle.
@app.post("/api/system/activate_cycle", response_model=SystemData)
def activate_cycle():
    with system.lock:
        system.active_event.set()  # Set the event to resume the simulation.
        return SystemData(
            manual_override=system.manual_override,
            deactivated=(not system.active_event.is_set())
        )


# API endpoint to retrieve all alert messages.
@app.get("/api/alerts", response_model=List[str])
def get_alerts():
    with system.lock:
        return system.alerts  # Return the list of alert messages.


# API endpoint to clear the alert messages.
@app.post("/api/alerts/clear")
def clear_alerts():
    with system.lock:
        system.alerts = []  # Reset the alerts list.
        return {"message": "Alerts cleared"}  # Return a confirmation message.


# API endpoint to get aggregated consumption data based on a given period.
@app.get("/api/consumption", response_model=List[ConsumptionDataModel])
def get_consumption_data(period: str = Query("day", enum=["day", "month", "year"])):
    with system.lock:
        data = aggregate_consumption(system.consumption_log, period)  # Aggregate consumption data.
    return data  # Return the aggregated consumption data.


# New API endpoint to generate and return a line graph of water consumption over time.
@app.get("/api/consumption_chart")
def consumption_chart(period: str = Query("day", enum=["day", "month", "year"])):
    with system.lock:
        data = aggregate_consumption(system.consumption_log, period)  # Get aggregated consumption data.

    # Extract time labels (x-axis) and consumption values (y-axis) from the data.
    times = [entry["time"] for entry in data]
    consumption_values = [entry["consumption"] for entry in data]

    # Create a new figure for the line graph with specified dimensions.
    plt.figure(figsize=(10, 6))
    # Plot the consumption values as a line graph with markers.
    plt.plot(times, consumption_values, marker="o", linestyle="-", color="blue")
    plt.title("Water Consumption Over Time")  # Set the title of the graph.
    plt.xlabel("Time")  # Label the x-axis.
    plt.ylabel("Consumption")  # Label the y-axis.
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability.
    plt.tight_layout()  # Adjust the layout so labels fit well.

    # Save the plot into a BytesIO buffer as a PNG image.
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()  # Close the plot to free resources.
    buf.seek(0)  # Rewind the buffer to the beginning.
    # Return the image as a streaming response with the appropriate media type.
    return StreamingResponse(buf, media_type="image/png")


# Create the initial simulation system with three tanks having capacities 100, 150, and 120 respectively.
system = MultiTankSystem([Tank(100), Tank(150), Tank(120)], simulation_speed=1.0)
# Start the simulation thread to begin the simulation loop.
start_simulation()
