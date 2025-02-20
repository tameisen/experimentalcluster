import subprocess
import time
import sys

def read_vcgencmd_output():
    # Führt den Befehl `vcgencmd pmic_read_adc` aus und liest den Output
    result = subprocess.run(['vcgencmd', 'pmic_read_adc'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

def parse_output(data):
    # Zeitstempel in Nanosekunden
    timestamp = int(time.time() * 1e9)
    
    # Output vorbereiten
    output = []

    # Zeilen parsen und in InfluxDB Line Protocol Format umwandeln
    for line in data.strip().split('\n'):
        key_value = line.split('=')
        if len(key_value) == 2:
            key = key_value[0].strip().replace(' ', '_').replace('(', '').replace(')', '')
            value = key_value[1].strip().replace('A', '').replace('V', '')
            if "current" in key:
                measurement = "current"
            elif "volt" in key:
                measurement = "voltage"
            else:
                continue

            output.append(f"{measurement},sensor={key} value={value} {timestamp}")

    return output

def main():
    # Warten auf Input von STDIN, um das Script zu triggern
    for line in sys.stdin:
        if line.strip() == "":
            # Wenn eine leere Zeile empfangen wird, wird das Script ausgeführt

            # Output von vcgencmd lesen
            data = read_vcgencmd_output()

            # Ausgabe parsen und formatieren
            formatted_output = parse_output(data)

            # Ausgabe für Telegraf/InfluxDB
            for line in formatted_output:
                print(line)

if __name__ == "__main__":
    main()
