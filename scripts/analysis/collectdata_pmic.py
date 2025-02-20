import re
import pandas as pd
import pytz
from influxdb_client import InfluxDBClient
from sqlalchemy import create_engine


# -------------------------------------------------------------------
# 0. NodeList aufsplitten
# -------------------------------------------------------------------
def expand_nodelist(node_str):
    """
    Zerlegt einen NodeList-String (z.B. "node[001-002]", "node001,node002")
    in einzelne Nodes.
    Beispiel:
      - "node[002-003]"  -> ["node002", "node003"]
      - "node[001-002],node010" -> ["node001","node002","node010"]
      - "node003" -> ["node003"]
    """
    results = []
    # Schritt 1: Komma-getrennte Blöcke aufteilen
    parts = node_str.split(',')
    for part in parts:
        part = part.strip()
        # Regex für node[xxx-yyy]
        match = re.match(r'^(?P<prefix>.*)\[(?P<start>\d+)-(?P<end>\d+)\]$', part)
        if match:
            prefix = match.group('prefix')  # z.B. "node"
            start_num = int(match.group('start'))
            end_num = int(match.group('end'))
            for num in range(start_num, end_num + 1):
                results.append(f"{prefix}{num:03d}")
        else:
            # Kein Bereich -> direkter Hostname
            results.append(part)
    return results


# -------------------------------------------------------------------
# 1. InfluxDB-Abfrage
# -------------------------------------------------------------------
def get_influx_data(url, token, org, query):
    """
    Fragt InfluxDB 2.x ab und gibt EIN DataFrame zurück,
    auch wenn mehrere Tabellen (df_list) geliefert werden.
    """
    try:
        client = InfluxDBClient(url=url, token=token, org=org)
        df_list = client.query_api().query_data_frame(query)
    except Exception as e:
        print(f"Fehler beim Abfragen der InfluxDB: {e}")
        return pd.DataFrame()
    finally:
        client.close()

    # Falls mehrere DataFrames, zusammenführen
    if isinstance(df_list, list):
        if len(df_list) == 0:
            return pd.DataFrame()
        elif len(df_list) == 1:
            df = df_list[0]
        else:
            df = pd.concat(df_list, ignore_index=True)
    else:
        df = df_list

    # Optional: Zeitspalte zu Datetime
    if "_time" in df.columns:
        df["_time"] = pd.to_datetime(df["_time"])

    return df


# -------------------------------------------------------------------
# 2. Slurm-DB-Abfrage (lokaler Socket)
# -------------------------------------------------------------------
def get_slurm_data_local_socket(db_user, db_name, query, socket_path="/var/run/mysqld/mysqld.sock"):
    """
    Liest über einen Unix-Socket (z.B. /var/run/mysqld/mysqld.sock)
    aus einer MySQL/MariaDB (Slurm-DB).
    """
    connection_str = f"mysql+mysqlconnector://{db_user}@localhost/{db_name}?unix_socket={socket_path}"

    try:
        engine = create_engine(connection_str)
        df = pd.read_sql_query(query, con=engine)
    except Exception as e:
        print(f"Fehler beim Abfragen der Slurm-DB: {e}")
        return pd.DataFrame()

    local_tz = pytz.timezone("Europe/Berlin")  # Oder deine lokale Zeitzone

    if "start_time" in df.columns:
        # 1) parse zu datetime (noch "naiv", ohne tz)
        df["start_time"] = pd.to_datetime(df["start_time"])
        # 2) lokalisieren -> "start_time ist in local_tz"
        df["start_time"] = df["start_time"].dt.tz_localize(local_tz, ambiguous='infer', nonexistent='shift_forward')
        # 3) nach UTC konvertieren
        df["start_time"] = df["start_time"].dt.tz_convert("UTC")

    if "end_time" in df.columns:
        df["end_time"] = pd.to_datetime(df["end_time"])
        df["end_time"] = df["end_time"].dt.tz_localize(local_tz, ambiguous='infer', nonexistent='shift_forward')
        df["end_time"] = df["end_time"].dt.tz_convert("UTC")

    return df

# -------------------------------------------------------------------
# 3. Hilfsfunktion zum verkürzen der jobbnamen
# -------------------------------------------------------------------
def shorten_job_name(job_name):
    """
    Schneidet das letzte '_xxxxxxxx' (8 Zeichen + Underscore) ab,
    falls der Jobname mindestens 9 Zeichen mehr als "_xxxxxxxx" hat.
    """
    if len(job_name) > 8 and re.match(r'.*_[0-9a-fA-F]{8}$', job_name):
        return job_name[:-9]  # entfernt "_ + 8 Zeichen" am Ende
    else:
        return job_name

# -------------------------------------------------------------------
# 4. Hauptprogramm
# -------------------------------------------------------------------
def main():
    # ---------------------------
    # A) InfluxDB-Parameter
    # ---------------------------
    influx_url = "http://192.168.113.20:8086"
    influx_token = "WydqpfRnLQ2br7jXQva1m3GIey31llPEA5T9zOy72w0="
    influx_org = "clusterwatch"

    # Komplette Filter & Berechnung für "pmic_metrics"
    # (Kein range, kein host-Filter – das fügen wir dynamisch an)
    base_influx_query = """
  |> filter(fn: (r) => exists r["host"])
  |> filter(fn: (r) => r["_measurement"] == "pmic_metrics")
  |> filter(fn: (r) => r["sensor"] == "0V8_AON_A_current16" or 
                       r["sensor"] == "0V8_AON_V_volt19" or 
                       r["sensor"] == "0V8_SW_A_current6" or 
                       r["sensor"] == "VDD_CORE_V_volt15" or 
                       r["sensor"] == "VDD_CORE_A_current7" or 
                       r["sensor"] == "HDMI_V_volt23" or 
                       r["sensor"] == "HDMI_A_current22" or 
                       r["sensor"] == "DDR_VDDQ_V_volt12" or 
                       r["sensor"] == "EXT5V_V_volt24" or 
                       r["sensor"] == "DDR_VDDQ_A_current4" or 
                       r["sensor"] == "DDR_VDD2_V_volt11" or 
                       r["sensor"] == "DDR_VDD2_A_current3" or 
                       r["sensor"] == "BATT_V_volt25" or 
                       r["sensor"] == "3V7_WL_SW_V_volt8" or 
                       r["sensor"] == "3V7_WL_SW_A_current0" or 
                       r["sensor"] == "3V3_SYS_V_volt9" or 
                       r["sensor"] == "3V3_SYS_A_current1" or 
                       r["sensor"] == "3V3_DAC_V_volt20" or 
                       r["sensor"] == "3V3_DAC_A_current17" or 
                       r["sensor"] == "3V3_ADC_V_volt21" or 
                       r["sensor"] == "3V3_ADC_A_current18" or 
                       r["sensor"] == "1V8_SYS_V_volt10" or 
                       r["sensor"] == "1V8_SYS_A_current2" or 
                       r["sensor"] == "1V1_SYS_V_volt13" or 
                       r["sensor"] == "1V1_SYS_A_current5" or 
                       r["sensor"] == "0V8_SW_V_volt14")
  |> pivot(rowKey:["_time", "host"], columnKey: ["sensor"], valueColumn: "_value")
  |> map(fn: (r) => ({
      _time: r._time,
      host: r.host,
      _field: "power",
      // Hier die Summierung aller Sensoren:
      _value: (if exists r["0V8_AON_V_volt19"] and exists r["0V8_AON_A_current16"] then 
                float(v: r["0V8_AON_V_volt19"]) * float(v: r["0V8_AON_A_current16"]) else 0.0) +
              (if exists r["0V8_SW_V_volt14"] and exists r["0V8_SW_A_current6"] then 
                float(v: r["0V8_SW_V_volt14"]) * float(v: r["0V8_SW_A_current6"]) else 0.0) +
              (if exists r["VDD_CORE_V_volt15"] and exists r["VDD_CORE_A_current7"] then 
                float(v: r["VDD_CORE_V_volt15"]) * float(v: r["VDD_CORE_A_current7"]) else 0.0) +
              (if exists r["HDMI_V_volt23"] and exists r["HDMI_A_current22"] then 
                float(v: r["HDMI_V_volt23"]) * float(v: r["HDMI_A_current22"]) else 0.0) +
              (if exists r["DDR_VDDQ_V_volt12"] and exists r["DDR_VDDQ_A_current4"] then 
                float(v: r["DDR_VDDQ_V_volt12"]) * float(v: r["DDR_VDDQ_A_current4"]) else 0.0) +
              (if exists r["DDR_VDD2_V_volt11"] and exists r["DDR_VDD2_A_current3"] then 
                float(v: r["DDR_VDD2_V_volt11"]) * float(v: r["DDR_VDD2_A_current3"]) else 0.0) +
              (if exists r["BATT_V_volt25"] then 
                float(v: r["BATT_V_volt25"]) * 0.0 else 0.0) +  // Annahme: Kein Stromwert für BATT_V
              (if exists r["3V7_WL_SW_V_volt8"] and exists r["3V7_WL_SW_A_current0"] then 
                float(v: r["3V7_WL_SW_V_volt8"]) * float(v: r["3V7_WL_SW_A_current0"]) else 0.0) +
              (if exists r["3V3_SYS_V_volt9"] and exists r["3V3_SYS_A_current1"] then 
                float(v: r["3V3_SYS_V_volt9"]) * float(v: r["3V3_SYS_A_current1"]) else 0.0) +
              (if exists r["3V3_DAC_V_volt20"] and exists r["3V3_DAC_A_current17"] then 
                float(v: r["3V3_DAC_V_volt20"]) * float(v: r["3V3_DAC_A_current17"]) else 0.0) +
              (if exists r["3V3_ADC_V_volt21"] and exists r["3V3_ADC_A_current18"] then 
                float(v: r["3V3_ADC_V_volt21"]) * float(v: r["3V3_ADC_A_current18"]) else 0.0) +
              (if exists r["1V8_SYS_V_volt10"] and exists r["1V8_SYS_A_current2"] then 
                float(v: r["1V8_SYS_V_volt10"]) * float(v: r["1V8_SYS_A_current2"]) else 0.0) +
              (if exists r["1V1_SYS_V_volt13"] and exists r["1V1_SYS_A_current5"] then 
                float(v: r["1V1_SYS_V_volt13"]) * float(v: r["1V1_SYS_A_current5"]) else 0.0)
    }))
  |> yield(name: "raw")
"""

    # ---------------------------
    # B) SlurmDB: 100 neueste Jobs
    # ---------------------------

    slurm_query = """
    SELECT
        id_job AS job_id,
        job_name AS job_name,
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

    # ---------------------------
    # C) Schleife: pro Job -> expand_nodelist -> pro Node
    # ---------------------------
    results = []

    for _, job in slurm_data.iterrows():
        job_id     = job["job_id"]
        original_name   = job["job_name"]
        node_str   = job["node"]  # z.B. "node[002-003]"
        start_time = job["start_time"]
        end_time   = job["end_time"]

        job_name = shorten_job_name(original_name)

        # Skip, wenn Zeiten ungültig
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

        # NodeList expandieren (z.B. "node[001-002]" -> ["node001","node002"])
        expanded_nodes = expand_nodelist(node_str)
        print(f"JOB {job_id} ({job_name}) RAW node_str={node_str}, expanded={expanded_nodes}")
        # Für jeden Node einzeln abfragen
        for single_node in expanded_nodes:
            # Dynamische Zeitangaben
            start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_str   = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Flux-Query zusammenbauen
            flux_query = f"""
from(bucket: "clusterdata")
  |> range(start: time(v: "{start_str}"), stop: time(v: "{end_str}"))
  |> filter(fn: (r) => r["host"] == "{single_node}")
{base_influx_query}
"""

            # 1) Daten holen
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

            # 2) Zeitintegration in Python:
            #    Wir interpretieren _value als Momentanleistung (W).
            #    -> Summieren die Energie in Joule = W * s
            df_influx = df_influx.sort_values(by="_time")

            # Zeitdifferenz in Sekunden
            df_influx["_delta_s"] = df_influx["_time"].diff().dt.total_seconds().fillna(0)
            # Energie pro Intervall = W * s
            df_influx["_energy_j"] = df_influx["_value"] * df_influx["_delta_s"]
            total_energy_j = df_influx["_energy_j"].sum()

            # Durchschnittsleistung (über gesamten Zeitraum)
            runtime_s = (df_influx["_time"].iloc[-1] - df_influx["_time"].iloc[0]).total_seconds()
            if runtime_s > 0:
                avg_power_w = total_energy_j / runtime_s
            else:
                avg_power_w = 0.0

            results.append({
                "job_id": job_id,
                "job_name": job_name,
                "node": single_node,
                "start_time": start_time,
                "end_time": end_time,
                "total_energy_j": total_energy_j,
                "avg_power_w": avg_power_w
            })

    # ---------------------------
    # D) Ergebnisse
    # ---------------------------
    final_df = pd.DataFrame(results)
    print(final_df)

    out_csv = "job_consumption_per_node_pmic.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"Die Verbrauchsdaten (pro Node & Job) wurden in '{out_csv}' gespeichert.")


if __name__ == "__main__":
    main()
