import traci

sumoBinary = r"D:\Eclipse\Sumo\bin\sumo-gui.exe"  # full path to SUMO GUI
sumoConfig = r"C:\Users\HP\Documents\SmartTraffic\simple.sumocfg"

traci.start([sumoBinary, "-c", sumoConfig])

# Get traffic light IDs
tls_ids = traci.trafficlight.getIDList()
print("Traffic lights:", tls_ids)

# Get phase count using RedYellowGreenDefinition
def get_phase_count(tls_id):
    # returns number of phases for this traffic light
    return len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases)

# Simple loop to switch traffic lights
for step in range(100):
    traci.simulationStep()  # advance 1 second

    for tls_id in tls_ids:
        current_phase = traci.trafficlight.getPhase(tls_id)
        phase_count = get_phase_count(tls_id)
        print(f"Step {step}, TLS {tls_id} current phase: {current_phase}")

        # Switch phase every 10 steps
        if step % 10 == 0:
            next_phase = (current_phase + 1) % phase_count
            traci.trafficlight.setPhase(tls_id, next_phase)

traci.close()
