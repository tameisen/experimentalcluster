import subprocess
import time
import sys

def read_vcgencmd_output():
    # Runs `vcgencmd pmic_read_adc` and reads the output
    result = subprocess.run(['vcgencmd', 'pmic_read_adc'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

def parse_output(data):
    # Timestamp in Nanoseconds
    timestamp = int(time.time() * 1e9)
    

    output = []

    # parsing step and transformation to the influxdb line protocol
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
    # Wait for Input from STDIN to trigger
    for line in sys.stdin:
        if line.strip() == "":
            # If empty input, read and parse

            data = read_vcgencmd_output()

            formatted_output = parse_output(data)

            for line in formatted_output:
                print(line)

if __name__ == "__main__":
    main()
