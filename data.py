from pathlib import Path
import zipfile, re
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
RAW = SCRIPT_DIR.parent / "data"
PROC = RAW / "processed"
GBFS = "https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information"

def fetch_stations():
    """
    Getting all the station info via api if the sattions.csv doesn't exist
    """
    RAW.mkdir(parents=True, exist_ok=True)
    r = requests.get(GBFS, timeout=30); r.raise_for_status()
    x = pd.DataFrame(r.json()["data"]["stations"])
    x = x.rename(columns={"station_id":"station_id"})
    x["station_id"] = x.station_id.astype(str)
    x.to_csv(RAW / "stations.csv", index=False)
    return x

def safe_read_csv(filepath_or_buffer, **kwargs):
    """
    Attempts to read a CSV with UTF-8 first, falling back to Latin-1 on decode failure.
    """
    try:
        return pd.read_csv(filepath_or_buffer, **kwargs)
    except UnicodeDecodeError:
        if isinstance(filepath_or_buffer, Path):
            return pd.read_csv(filepath_or_buffer, encoding="latin-1", **kwargs)
        elif hasattr(filepath_or_buffer, "seek"):
            filepath_or_buffer.seek(0)
            return pd.read_csv(filepath_or_buffer, encoding="latin-1", **kwargs)
        raise

def load_trips():
    """
    Finds and loops through all the bikeshare CSV files stored inside the data folder. It 
    strips out extra space or capitalization variants from headers before loading it.
    """
    frames = []
    for p in RAW.rglob("*"):
        if p.suffix.lower() == ".csv":
            if p.name == "stations.csv":
                continue
            frames.append(safe_read_csv(p, low_memory=False))
                            
    if not frames: 
        raise FileNotFoundError(f"Put downloaded Toronto ridership CSVfiles in: {RAW}")
        
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip().str.lower()
    
    needed = {"start_time", "start_station_id", "end_time","end_station_id"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Could not identify {needed}. Dataset columns: {list(df.columns)}")
        
    df["start_time"] = pd.to_datetime(df.start_time, errors="coerce")
    df["end_time"] = pd.to_datetime(df.end_time, errors="coerce")
    for c in ["start_station_id", "end_station_id"]: 
        df[c] = df[c].astype(str).str.replace(r"\.0$", "", regex=True).astype("string")

    return df.dropna(subset=["start_time", "start_station_id","end_time","end_station_id"])


def prepare(start="2025-07-01", end="2025-10-01", n_stations=50):
    """
    This method cuts down and processes the dataset into cleaned data. It filters trips down to 
    a specific date range, isolates the top N busiest stations to manage state space bounds, 
    extracts hourly/weekday timestamps and groups historical rides to compute the average 
    customer request rates per hour.
    Also sets the capacity of all stations at 25 if not given in the stations.csv and if a station
    id can't be found, it makes it equal to the start/dest station so we assume the user brought it back.
    The reason for this was because I've seen data that had unkown station_id's and google said toronto
    bikeshare does that for this reason. Lowkey buns.
    """
    PROC.mkdir(parents=True, exist_ok=True)
    
    if not (RAW / "stations.csv").exists(): 
        fetch_stations()
    st_registry = safe_read_csv(RAW / "stations.csv")
    valid_ids = set(st_registry["station_id"].astype(str).tolist())
    
    trips = load_trips()
    trips["start_station_id"] = trips.start_station_id.astype(str)
    trips["end_station_id"] = trips.end_station_id.astype(str)

    #Checking valid station id's sometimes they use placeholders that dont exist when the user brings back the bike
    #to the same location
    start_unknown = ~trips["start_station_id"].isin(valid_ids)
    trips.loc[start_unknown, "start_station_id"] = trips.loc[start_unknown, "end_station_id"]
    
    end_unknown = ~trips["end_station_id"].isin(valid_ids)
    trips.loc[end_unknown, "end_station_id"] = trips.loc[end_unknown, "start_station_id"]
    
    trips = trips[trips["start_station_id"].isin(valid_ids) & trips["end_station_id"].isin(valid_ids)].copy()
    
    trips = trips[(trips.start_time >= start) & (trips.start_time < end)].copy()
    if trips.empty:
        dates = load_trips().start_time.agg(["min", "max"])
        raise ValueError(f"No trips in requested window {start} to {end}. Available data bounds: {dates['min']} to {dates['max']}.")
        
    counts = trips.start_station_id.value_counts().add(trips.end_station_id.value_counts(), fill_value=0)
    ids = counts.nlargest(n_stations).index.astype(str).tolist()
    trips = trips[trips.start_station_id.isin(ids) & trips.end_station_id.isin(ids)].copy()
    
    st = st_registry.copy()
    st["station_id"] = st.station_id.astype(str)
    st = st[st.station_id.isin(ids)].set_index("station_id").reindex(ids).reset_index()
    
    if st[["lat", "lon"]].isna().any().any(): 
        missing_coords = st[st["lat"].isna() | st["lon"].isna()]
        print(missing_coords[["station_id", "name"]])
        raise ValueError("Missing coordinates for selected stations.")
        
    if "capacity" not in st: 
        st["capacity"] = 25
    st["capacity"] = pd.to_numeric(st.capacity, errors="coerce").fillna(25).clip(lower=5)
    
    trips["hour"] = trips.start_time.dt.hour
    trips["weekday"] = trips.start_time.dt.dayofweek
    trips.to_parquet(PROC / "trips.parquet", index=False)
    st.to_csv(PROC / "stations.csv", index=False)
    
    od = trips.groupby(["weekday", "hour", "start_station_id", "end_station_id"]).size().rename("count").reset_index()
    days = trips.assign(day=trips.assign(day=trips.start_time.dt.date).drop_duplicates("day").groupby(trips.assign(day=trips.start_time.dt.date).drop_duplicates("day").start_time.dt.dayofweek).size())
    od["count"] = od.apply(lambda r: r["count"] / max(int(days.get(int(r["weekday"]), 1)), 1), axis=1)
    od.to_parquet(PROC / "od_demand.parquet", index=False)
    
    print(f"Prepared {len(trips):,} trips, {len(st)} stations. With dates from: {trips.start_time.min()} to {trips.start_time.max()}")
    return trips, st, od

if __name__ == "__main__": 
    prepare()