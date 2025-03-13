import sys
import serial
import time
from serial.tools import list_ports
import skrf as rf  # scikit-rf  Requires typing_extensions
import numpy as np

# NanoVNA USB IDs
VNA_VID = 0x0483
VNA_PID = 0x5740

# Initialize global variables
GUImessage = ""
VNA_version_Info = ""

# Detect NanoVNA Port
def VNAgetPort() -> str:
    global GUImessage
    GUImessage = "Scanning USB Ports for VNA"
    print("[INFO] Scanning USB Ports for VNA")
    device_list = list_ports.comports()
    for device in device_list:
        if device.vid is not None and device.pid is not None:
            print(f"     Device {device.device} has VID {device.vid:04X} : PID {device.pid:04X}")
        else:
            print(f"     Device {device.device} has no VID or PID")
        if device.vid == VNA_VID and device.pid == VNA_PID:
            print(f"[INFO] Found NanoVNA on port: {device.device}")
            GUImessage = f"NanoVNA device detected on port {device}"
            return device.device
    GUImessage = "No NanoVNA device detected"
    raise OSError("No NanoVNA device detected")
# Init NanoVNA Com Port
def VNAserialComInit(port: str, baudrate=38400, timeout=5):
    global GUImessage
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"[INFO] VNA connected to {port} at {baudrate} baud.")
        GUImessage = f"NanoVNA device connected to port {port}"
        ser.flush()
        return ser
    except serial.SerialException as e:
        print(f"[ERROR] Error opening serial port: {e}")
        return None

# Connect or reconnect to NanoVNA
def VNAconnectOrReconnect():
    global ser, VNA_version_Info
    retry_delay = 1
    try:
        ser.close()
    except NameError:
        pass  # Do nothing if 'ser' is not defined
    while True:
        try:
            port = VNAgetPort()  # will raise an error if not found
            ser = VNAserialComInit(port)

            if ser:
                # clear VNA buffer
                response = ""
                command = "\r"
                responseTest = 0
                while not "ch" in response:
                    ser.write(command.encode('utf-8'))
                    time.sleep(0.2)  # Adjust sleep time as needed
                    response = ser.read_all().decode('utf-8')
                    # print("A raw for debug only:", repr(response))
                    responseTest = responseTest + 1
                    if responseTest > 10:
                        print("❌ [ERROR] No response from VNA, unplug & replug")

                print("[INFO] VNA reset")
                command = "reset\r"
                ser.write(command.encode('utf-8'))
                ser.close()
                time.sleep(1)
                print("[INFO] VNA re-connection")
                ser = VNAserialComInit(port)
                command = "\r"
                responseTest = 0
                while not "ch" in response:
                    ser.write(command.encode('utf-8'))
                    time.sleep(0.2)  # Adjust sleep time as needed
                    response = ser.read_all().decode('utf-8')
                    # print("A raw for debug only:", repr(response))
                    responseTest = responseTest + 1
                    if responseTest > 10:
                        print("❌ [ERROR] No response from VNA, unplug & replug")

                response = ""
                command = "\r"
                while not "ch" in response:
                    ser.write(command.encode('utf-8'))
                    time.sleep(0.2)  # Adjust sleep time as needed
                    response = ser.read_all().decode('utf-8')
                    # print("A raw for debug only:", repr(response))
                print("[INFO] Getting VNA firmware version")

                command = "version\r"
                while not ('version' in response and 'ch>' in response):
                    ser.write(command.encode('utf-8'))
                    time.sleep(.2)
                    response = ser.read_all().decode('utf-8')
                    # print("AX raw for debug only:", repr(response))
                    time.sleep(0.2)

                # Extract version info using split
                lines = response.split('\r\n')
                VNA_version_Info = lines[1] if len(lines) > 1 else None
                # print(f"Received response: {response}")
                print(f"[INFO] VNA version info: {VNA_version_Info}")
                print(f"✅[INFO] VNA now Ready to Receive Commands")
                return
            else:
                print("[ERROR] Failed to open serial connection.")
        except Exception as e:
            print(f"[ERROR] {e}")
        print(f"[WARNING] Retrying connection in {retry_delay} seconds...")
        time.sleep(retry_delay)

    print("[ERROR] Reconnection failed.")

def VNAexecCommand(cmd, timeout=2):
    """Send command to NanoVNA and read response until 'ch>' appears."""
    if ser is None or not ser.is_open:
        print("[ERROR] Serial connection lost. Attempting to reconnect...")
        return ""

    try:
        ser.write(cmd.encode())  # Send to NanoVNA
        time.sleep(0.05)  # ✅ Small delay after sending command
        response = ""
        start_time = time.time()

        while True:
            if ser is None or not ser.is_open:
                print("[ERROR] Serial connection lost during command execution.")
                break

            try:
                if ser.in_waiting > 0:
                    response += ser.read(ser.in_waiting).decode('utf-8')
                    time.sleep(0.01)  # ✅ Small delay after reading data

                    # ✅ Stop if 'ch>' is found at the end (end of data)
                    if "ch>" in response:
                        break
            except OSError as e:
                print(f"[ERROR] Serial device error: {e}")
                break

            # ✅ Break on timeout to avoid infinite loop
            if time.time() - start_time > timeout:
                print(f"[ERROR] Timeout waiting for response from VNA after {timeout} seconds")

                # ✅ Small delay before retrying reset
                time.sleep(0.1)

                command = "reset\r"
                ser.write(command.encode())
                time.sleep(0.1)  # ✅ Give it time to process the reset

                command = "reset\r"
                ser.write(command.encode())
                time.sleep(0.5)  # ✅ Allow VNA to reboot properly

                VNAconnectOrReconnect()
                return "reconnected"

            # ✅ Small sleep to reduce CPU load
            time.sleep(0.01)

    except (serial.SerialException, OSError) as e:
        print(f"[ERROR] Serial communication error: {e}")
        VNAconnectOrReconnect()
        return "reconnected"

    return response

def sweep_to_non_calibrated_network(start_freq, end_freq, points):
    global ser
    MAX_POINTS = 100  # tested with 400 for NanoVNA H2 but not faster

    print(f"\n[INFO] Sweeping from {start_freq:,} Hz to {end_freq:,} Hz with max points: {MAX_POINTS:,}")

    step_size = (end_freq - start_freq) // (points - 1)
    frequency_chunks = []
    current_start = start_freq
    total_points = 0

    # Split frequency range into chunks based on MAX_POINTS
    while total_points < points:
        chunk_stop = min(current_start + step_size * (MAX_POINTS - 1), end_freq)

        if chunk_stop == end_freq:
            chunk_points = points - total_points
        else:
            chunk_points = min((chunk_stop - current_start) // step_size + 1, MAX_POINTS)

        frequency_chunks.append((current_start, chunk_stop, chunk_points))
        total_points += chunk_points
        current_start = chunk_stop + step_size

        if total_points >= points:
            break

    all_frequencies = []
    all_s11_data = []
    start_time = time.time()

    # Process each chunk
    chunk_number = 0
    for chunk_start, chunk_stop, chunk_points in frequency_chunks:
        time.sleep(.6)
        chunk_number += 1
        print(f"   --> Sweep chunk n°{chunk_number} : from {chunk_start:,} Hz to {chunk_stop:,} Hz  ({chunk_points})")

        # Send the scan command for the current chunk
        command = f"scan {chunk_start} {chunk_stop} {chunk_points} 3\r"
        # print(f"[DEBUG] Sending command: {command.strip()}")
        response = VNAexecCommand(command)




        # ✅ Check for empty response or timeout
        if "reconnected" in response:
            return None

        # Debugging the raw response
        # print(f"[DEBUG] Raw response:\n{response}")
        # input("Press Enter to continue....")
        # Split lines and remove leading/trailing spaces
        lines = response.strip().splitlines()

        # Ignore the first line (command echo)
        if lines and lines[0].startswith('scan'):
            lines = lines[1:]

        for line in lines:
            if 'ch>' in line:  # Stop when end marker is received
                break

            # print(f"[DEBUG] Parsing line: {line}")
            try:
                freq, real, imag = map(float, line.split())
                if chunk_start <= freq <= chunk_stop:
                    all_frequencies.append(freq)
                    all_s11_data.append(complex(real, imag))
                else:
                    print(f"⚠️ [WARNING] Ignoring out-of-range frequency: {freq}")
            except ValueError:
                print(f"⚠️ [WARNING] Invalid data format: {line.strip()}")
                continue


    # ✅ Adjust the number of points to match the requested count
    if len(all_frequencies) > points:
        all_frequencies = all_frequencies[:points]
        all_s11_data = all_s11_data[:points]
    elif len(all_frequencies) < points:
        missing_points = points - len(all_frequencies)
        last_freq = all_frequencies[-1] if all_frequencies else start_freq
        step = step_size
        for _ in range(missing_points):
            last_freq += step
            all_frequencies.append(last_freq)
            all_s11_data.append(all_s11_data[-1] if all_s11_data else complex(0, 0))

    # ✅ After collecting all data, prepare it for scikit-rf Network
    if all_frequencies and all_s11_data:
        try:
            frequency_obj = rf.Frequency.from_f(all_frequencies, unit='Hz')
            s_parameters = np.array(all_s11_data).reshape((-1, 1, 1))
            network = rf.Network(frequency=frequency_obj, s=s_parameters)
            network.name = "non calibrated"
            # ✅ Save to Touchstone file for validation
            # network.write_touchstone("sweep_data.s1p")

            elapsed_time = (time.time() - start_time) * 1000
            print(f"✅{network} successfully created in {elapsed_time:,.0f} ms")

            return network
        except Exception as e:
            print(f"[ERROR] Failed to create network: {e}")
            return None
    else:
        print("[ERROR] No valid data collected.")
        return None

if __name__ == "__main__":
    VNAconnectOrReconnect()
    import random
    while True:
        start_freq = random.randint(5_000_000, 8_000_000)
        x = random.randint(1_000_000, 8_000_000)
        end_freq = start_freq + x
        points = random.randint(100, 400)
        sweep_to_non_calibrated_network(start_freq, end_freq, points)
