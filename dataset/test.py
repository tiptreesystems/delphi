import json
import statistics

# Load the forecasts file
with open("data/2024-07-21.ForecastBench.human_super_individual.json") as f:
    data = json.load(f)

target_id = "TPkEjiNb1wVCIGFnPcDD"
matching = [f["forecast"] for f in data["forecasts"] if f["id"] == target_id]

average = sum(matching) / len(matching) if matching else None
median = statistics.median([f["forecast"] for f in data["forecasts"] if f["id"] == "TPkEjiNb1wVCIGFnPcDD"])

print(f"Found {len(matching)} forecasts for ID {target_id}")
print(f"Average forecast: {average}")
print(f"Median forecast: {median}")
