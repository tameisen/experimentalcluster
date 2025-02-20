#!/usr/bin/env python3
import re
import pandas as pd
import pytz
from influxdb_client import InfluxDBClient
from sqlalchemy import create_engine

def get_influx_data(url, token, org, query):
    try:
        client = InfluxDBClient(url=url, token=token, org=org)
        df_list = client.query_api().query_data_frame(query)
    except Exception as e:
        print(f"Fehler beim Abfragen der InfluxDB: {e}")
        return pd.DataFrame()
    finally:
        client.close()

    if isinstance(df_list, list):
        if len(df_list) == 0:
            return pd.DataFrame()
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = df_list

    if "_time" in df.columns:
        df["_time"] = pd.to_datetime(df["_time"])
    return df

def get_slurm_data_local_socket(db_user, db_name, query, socket_path="/var/run/mysqld/mysqld.sock"):
    connection_str = f"mysql+mysqlconnector://{db_user}@localhost/{db_name}?unix_socket={socket_path}"
    try:
        engine = create_engine(connection_str)
        df = pd.read_sql_query(query, con=engine)
    except Exception as e:
        print(f"Fehler beim Abfragen der Slurm-DB: {e}")
        return pd.DataFrame()

    local_tz = pytz.timezone("Europe/Berlin")
    for col in ["start_time", "end_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df[col] = df[col].dt.tz_localize(local_tz, ambiguous='infer', nonexistent='shift_forward').dt.tz_convert("UTC")

    return df

def expand_nodelist(node_str):
    results = []
    parts = node_str.split(',')
    for part in parts:
        part = part.strip()
        match = re.match(r'^(?P<prefix>.*)\[(?P<start>\d+)-(?P<end>\d+)\]$', part)
        if match:
            prefix = match.group('prefix')
            start_num = int(match.group('start'))
            end_num = int(match.group('end'))
            for num in range(start_num, end_num + 1):
                results.append(f"{prefix}{num:03d}")
        else:
            results.append(part)
    return results

def shorten_job_name(job_name):
    """
    Schneidet das letzte '_xxxxxxxx' (8 Zeichen + Underscore) ab,
    falls der Jobname mindestens 9 Zeichen mehr als "_xxxxxxxx" hat.
    """
    if len(job_name) > 8 and re.match(r'.*_[0-9a-fA-F]{8}$', job_name):
        return job_name[:-9]  # entfernt "_ + 8 Zeichen" am Ende
    else:
        return job_name

def main():
    # Influx-Parameter
    influx_url = "http://192.168.113.20:8086"
    influx_token = "WydqpfRnLQ2br7jXQva1m3GIey31llPEA5T9zOy72w0="
    influx_org = "clusterwatch"

    # Neue Query-Vorgabe
    base_influx_query = """
    |> filter(fn: (r) => r["_measurement"] == "mqtt_consumer")
    |> filter(fn: (r) => r["_field"] == "aenergy_total")
    |> filter(fn: (r) => r["host"] == "{host_placeholder}")
    |> filter(fn: (r) => r["topic"] == "{topic_placeholder}/status/switch:0")
    |> yield(name: "raw")
    """

    # Beispiel-Slurm-Query
    slurm_query = """
    SELECT
        id_job AS job_id,
        job_name,
        nodelist AS node,
        FROM_UNIXTIME(time_start) AS start_time,
        FROM_UNIXTIME(time_end)   AS end_time
    FROM snowflake_job_table
    ORDER BY time_end DESC
    LIMIT 72;
    """

    slurm_user = "node"
    slurm_db = "slurm_acct_db"
    socket_path = "/var/run/mysqld/mysqld.sock"

    slurm_data = get_slurm_data_local_socket(
        db_user=slurm_user,
        db_name=slurm_db,
        query=slurm_query,
        socket_path=socket_path
    )
    if slurm_data.empty:
        print("Keine Jobs gefunden. Script endet.")
        return

    results = []

    for _, job in slurm_data.iterrows():
        job_id = job["job_id"]
        job_name = shorten_job_name(job.get("job_name", "no_name"))
        node_str = job["node"]
        start_time = job["start_time"]
        end_time = job["end_time"]

        if pd.isnull(start_time) or pd.isnull(end_time) or end_time <= start_time:
            results.append({
                "job_id": job_id,
                "job_name": job_name,
                "node": node_str,
                "start_time": start_time,
                "end_time": end_time,
                "total_energy_j": 0.0,
                "avg_power_w": 0.0
            })
            continue

        expanded_nodes = expand_nodelist(node_str)
        print(f"JOB {job_id} ({job_name}) RAW node_str={node_str}, expanded={expanded_nodes}")

        for single_node in expanded_nodes:
            start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_str   = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

            topic_placeholder = single_node.replace("node", "shelly")

            flux_query = f"""
from(bucket: "clusterdata")
  |> range(start: time(v: "{start_str}"), stop: time(v: "{end_str}"))
{base_influx_query.replace('{host_placeholder}', single_node).replace('{topic_placeholder}', topic_placeholder)}
"""

            df_influx = get_influx_data(
                url=influx_url,
                token=influx_token,
                org=influx_org,
                query=flux_query
            )

            if df_influx.empty:
                results.append({
                    "job_id": job_id,
                    "job_name": job_name,
                    "node": single_node,
                    "start_time": start_time,
                    "end_time": end_time,
                    "total_energy_j": 0.0,
                    "avg_power_w": 0.0
                })
                continue

            df_influx = df_influx.sort_values(by="_time")

            first_val = df_influx["_value"].iloc[0]
            last_val  = df_influx["_value"].iloc[-1]
            total_energy_j = (last_val - first_val) * 3600

            runtime_s = (df_influx["_time"].iloc[-1] - df_influx["_time"].iloc[0]).total_seconds()
            avg_power_w = total_energy_j / runtime_s if runtime_s > 0 else 0.0

            results.append({
                "job_id": job_id,
                "job_name": job_name,
                "node": single_node,
                "start_time": start_time,
                "end_time": end_time,
                "total_energy_j": total_energy_j,
                "avg_power_w": avg_power_w
            })

    final_df = pd.DataFrame(results)
    print(final_df)

    out_csv = "job_consumption_per_node_shelly.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"Die Verbrauchsdaten (pro Node & Job) wurden in '{out_csv}' gespeichert.")


if __name__ == "__main__":
    main()
