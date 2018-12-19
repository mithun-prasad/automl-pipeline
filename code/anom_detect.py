import argparse
from batch_scoring import *

parser = argparse.ArgumentParser("anom_detect")

parser.add_argument("--output_directory", type=str, help="output directory")
args = parser.parse_args()

print("output directory: %s" % args.output_directory)
os.makedirs(args.output_directory, exist_ok=True)

# public store of telemetry data
data_dir = 'https://sethmottstore.blob.core.windows.net/predmaint/'


print("Reading data ... ", end="")
telemetry = pd.read_csv(os.path.join(data_dir, 'telemetry.csv'))
print("Done.")

print("Parsing datetime...", end="")
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%m/%d/%Y %I:%M:%S %p")
print("Done.")

    
window_size = 12  # how many measures to include in rolling average
sensors = telemetry.columns[2:]
window_sizes = [window_size] * len(sensors)  # this can be changed to have individual window_sizes for each sensor
machine_ids = telemetry['machineID'].unique()

t = TicToc()
for machine_id in machine_ids:
    df = telemetry.loc[telemetry.loc[:, 'machineID'] == machine_id, :]
    t.tic()
    print("Working on sensor: ")
    for s, sensor in enumerate(sensors):
        N = window_sizes[s]
        print("   %s " % sensor)
        
        df_ra = rolling_average(df, sensor, N)
        anoms_timestamps = do_ad(df_ra)
        
        df_anoms = pd.DataFrame(data={'datetime': anoms_timestamps, 'machineID': [machine_id] * len(anoms_timestamps), 'errorID': [sensor] * len(anoms_timestamps)})
        
        # if this is the first machine and sensor, we initialize a new dataframe
        if machine_id == 1 and s == 0:
            df_anoms_all = df_anoms
        else: # otherwise we append the newly detected anomalies to the existing dataframe
            df_anoms_all = df_anoms_all.append(df_anoms, ignore_index=True)

    # store of output
    obj = {}
    obj["df_anoms"] = df_anoms_all

    out_file = os.path.join(args.output_directory, "anoms.pkl")
    with open(out_file, "wb") as fp:
        pickle.dump(obj, fp)
        
    t.toc("Processing machine %s took" % machine_id)


