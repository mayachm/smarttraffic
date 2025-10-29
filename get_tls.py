import traci

sumoBinary = r"D:\Eclipse\Sumo\bin\sumo-gui.exe"
  # or "sumo" if you want no GUI
sumoConfig = "simple.sumocfg"

# Start SUMO as TraCI server
traci.start([sumoBinary, "-c", sumoConfig])

# Get all traffic light IDs
tls_ids = traci.trafficlight.getIDList()
print("Traffic light IDs:", tls_ids)

# Optional: print the controlled lanes for each TLS
for tls_id in tls_ids:
    controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
    print(f"TLS {tls_id} controls lanes: {controlled_lanes}")

traci.close()
