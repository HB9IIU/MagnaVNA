
import os
import sys
import skrf as rf  # scikit-rf  Requires typing_extensions
import serial
import time
import numpy as np
from serial.tools import list_ports
import threading
from flask import Flask, request, jsonify, render_template
import logging
from scipy.signal import savgol_filter
import webbrowser
import traceback
import socket
global sweepNumber
global port
import platform
sweepNumber = 0
port=0
# Global Event to signal thread stop
stop_event = threading.Event()

global wideSweepOngoing
global narrowSweepOngoing
global recenteringwideSweepOngoing
wideSweepOngoing=False
narrowSweepOngoing=False
recenteringwideSweepOngoing=False


global calibrationFilesAvailable
calibrationFilesAvailable=False

global ws_frequency_array, ws_s11_db_array, ws_swr_array, ws_min_s11_db, ws_min_swr, ws_freq_at_min_s11, ws_freq_at_min_swr
global ns_frequency_array, ns_s11_db_array, ns_swr_array, ns_min_s11_db, ns_min_swr, ns_freq_at_min_s11, ns_freq_at_min_swr
global f1_2, f2_2, f1_3, f2_3, bw_2, bw_3
f1_2, f2_2, f1_3, f2_3, bw_2, bw_3 = 0, 0, 0, 0, 0, 0

ws_frequency_array =  None
ws_s11_db_array =   None
ws_swr_array =  None
ws_min_s11_db = None
ws_min_swr = None
ws_freq_at_min_s11 = None
ws_freq_at_min_swr =  None

ns_frequency_array =  None
ns_s11_db_array =   None
ns_swr_array =   None
ns_min_s11_db =  None
ns_min_swr =  None
ns_freq_at_min_s11 =  None
ns_freq_at_min_swr =  None


# NanoVNA USB IDs
VNA_VID = 0x0483  # Example VID for STM32-based devices (NanoVNA typically uses this)
VNA_PID = 0x5740  # NanoVNA PID

# NanoVNA USB IDs
VNA_version_Info=None


# Get the current system's OS
current_os = platform.system()
if current_os == "Darwin":
    print("You are on macOS, the land of stability and no blue screens! 😎")
    # Path to the calibration files
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    calibration_kit_dir = os.path.join(BASE_DIR, 'CalibrationKit')
    # Create directory if it do not exist
    os.makedirs(calibration_kit_dir, exist_ok=True)
    template_folder = os.path.join(BASE_DIR, 'templates')
    static_folder = os.path.join(BASE_DIR, 'static')
elif current_os == "Windows":
    print("You are on Windows... hope you don't see any blue screens!")
    # Determine the base directory where the executable or script is located
    if getattr(sys, 'frozen', False):  # Running as a bundled executable
        BASE_DIR = os.path.dirname(sys.executable)  # cx_Freeze uses this for the executable path
    else:  # Running as a script
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Paths for resources
    calibration_kit_dir = os.path.join(BASE_DIR, 'CalibrationKit')
    os.makedirs(calibration_kit_dir, exist_ok=True)

    template_folder = os.path.join(BASE_DIR, 'templates')
    static_folder = os.path.join(BASE_DIR, 'static')

    print("BASE_DIR:", BASE_DIR)
    print("CalibrationKit:", calibration_kit_dir)
    print("Templates:", template_folder)
    print("Static:", static_folder)
else:
    print(f"You are on {current_os}")
    sys.exit()

# --------------------------------------------------
# Flask Routes
# --------------------------------------------------
# Flask app initialization with explicit paths
app = Flask(
    __name__,
    template_folder=template_folder,  # Absolute path to templates
    static_folder=static_folder       # Absolute path to static files
)






@app.route('/')
def index_default():
    return render_template('index.html')

@app.route('/theory.html')
def theory():
    return render_template('theory.html')



@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/calibration.html')
def calibration():
    return render_template('calibration.html')

@app.route('/verification.html')
def verification():
    return render_template('verification.html')






@app.route('/goodbye.html')
def goodbye():
    return render_template('goodbye.html')

@app.route('/get_general_configuration_data')
def return_general_configuration_data():
    data = {
        'wide_Sweep_Start_Frequency': wide_Sweep_Start_Frequency,
        'wide_Sweep_Stop_Frequency': wide_Sweep_Stop_Frequency,
        'wide_Sweep_Number_Of_Points': wide_Sweep_Number_Of_Points,
        'VNA_version_Info': VNA_version_Info,
        'calibrationFilesAvailable': calibrationFilesAvailable
    }
    return jsonify(data)

@app.route('/wide_sweep_data')
def wide_sweep_data():
    if ws_frequency_array is None:
        return jsonify("")
    data = {
        'ws_frequency_array': ws_frequency_array.tolist(),
        'ws_s11_db_array': ws_s11_db_array.tolist(),
        'ws_min_s11_db': ns_min_s11_db,
        'ws_freq_at_min_s11': ns_freq_at_min_s11,
    }
    return jsonify(data)


@app.route('/zoom_data')
def zoom_data():
    if ns_frequency_array is None:
        return jsonify("")
    data = {
        'ns_frequency_array': ns_frequency_array.tolist(),
        'ns_s11_db_array': ns_s11_db_array.tolist(),
        'ns_swr_array': ns_swr_array.tolist(),
        'ns_min_s11_db': ns_min_s11_db,
        'ns_min_swr': ns_min_swr,
        'ns_freq_at_min_s11': ns_freq_at_min_s11,
        'f1_2': f1_2,
        'f2_2': f2_2,
        'f1_3': f1_3,
        'f2_3': f2_3,
        'bw_2': bw_2,
        'bw_3': bw_3,
        'recenteringwideSweepOngoing': recenteringwideSweepOngoing
    }
    return jsonify(data)




@app.route('/calibrate_short', methods=['POST'])
def calibrate_short():
    global calibrationFilesAvailable
    CalibratedNetwork = sweep_to_non_calibrated_network(calibration_start_freq, calibration_end_freq, calibration_points)
    save_network_to_file(CalibratedNetwork, "SHORT_cal")
    calibrationFilesAvailable=checkIfCalibrationFilesExist()
    return jsonify({'message': 'Short calibration completed successfully!'})

@app.route('/calibrate_open', methods=['POST'])
def calibrate_open():
    global calibrationFilesAvailable
    CalibratedNetwork = sweep_to_non_calibrated_network( calibration_start_freq, calibration_end_freq, calibration_points)
    save_network_to_file(CalibratedNetwork, "OPEN_cal")
    calibrationFilesAvailable=checkIfCalibrationFilesExist()
    return jsonify({'message': 'Open calibration completed successfully!'})

@app.route('/calibrate_load', methods=['POST'])
def calibrate_load():
    global calibrationFilesAvailable
    CalibratedNetwork = sweep_to_non_calibrated_network(calibration_start_freq, calibration_end_freq, calibration_points)
    save_network_to_file(CalibratedNetwork, "LOAD_cal")
    calibrationFilesAvailable=checkIfCalibrationFilesExist()
    return jsonify({'message': 'Load calibration completed successfully!'})

@app.route('/peform_wide_sweep', methods=['POST'])
def peform_wide_sweep():
    centering_wide_sweep();
    return jsonify({'message': 'Performing wide sweep!'})


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the Flask server and stop the sweeping thread."""
    global stop_event, sweep_thread, ser

    # Stop the sweeping thread if it's running
    if sweep_thread.is_alive():
        print("Stopping sweeping thread...")
        stop_event.set()  # Signal the sweeping thread to stop
        sweep_thread.join()  # Wait for the thread to finish
        print("Sweeping thread stopped.")

    # Close the serial connection if it's open
    if ser is not None and ser.is_open:
        print("Closing serial connection...")
        ser.close()
        print("Serial connection closed.")

    # Shut down the Flask server
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        print("Shutting down Flask server using Werkzeug...")
        func()
    else:
        print("Not running on Werkzeug; exiting the process...")

    # Exit the process explicitly

    os._exit(0)  # Force exit the process

@app.route('/stop_continuous_sweeping_thread', methods=['POST'])
def stop_continuous_sweeping_thread():
    """Shutdown the Flask server and stop the sweeping thread."""
    global stop_event, sweep_thread
    # Stop the sweeping thread if it's running
    if sweep_thread.is_alive():
        print("Stopping sweeping thread...")
        stop_event.set()  # Signal the sweeping thread to stop
        sweep_thread.join()  # Wait for the thread to finish
        print("Sweeping thread stopped.")
        return jsonify({'message': 'Sweeping thread stopped.!'})
    else:
        return jsonify({'message': 'Sweeping thread was already stopped.!'})



@app.route('/sweep_short', methods=['POST'])
def sweep_short():
    nonCalibratedNetwork=verify()
    print(nonCalibratedNetwork)
    #    Extract data from the Network object
    frequencies = nonCalibratedNetwork.f  # Frequencies in Hz
    s11_data = nonCalibratedNetwork.s[:, 0, 0]  # S11 parameter

    # Prepare the response data
    response_data = {
        "frequencies": list(frequencies),  # Convert numpy array to list
        "s11": [{"real": s.real, "imag": s.imag} for s in s11_data]
    }
    return jsonify(response_data), 200

@app.route('/sweep_open', methods=['POST'])
def sweep_open():
    nonCalibratedNetwork=verify()
    print(nonCalibratedNetwork)
    #    Extract data from the Network object
    frequencies = nonCalibratedNetwork.f  # Frequencies in Hz
    s11_data = nonCalibratedNetwork.s[:, 0, 0]  # S11 parameter

    # Prepare the response data
    response_data = {
        "frequencies": list(frequencies),  # Convert numpy array to list
        "s11": [{"real": s.real, "imag": s.imag} for s in s11_data]
    }
    return jsonify(response_data), 200

@app.route('/sweep_load', methods=['POST'])
def sweep_load():
    nonCalibratedNetwork=verify()
    print(nonCalibratedNetwork)
    #    Extract data from the Network object
    frequencies = nonCalibratedNetwork.f  # Frequencies in Hz
    s11_data = nonCalibratedNetwork.s[:, 0, 0]  # S11 parameter

    # Prepare the response data
    response_data = {
        "frequencies": list(frequencies),  # Convert numpy array to list
        "s11": [{"real": s.real, "imag": s.imag} for s in s11_data]
    }
    return jsonify(response_data), 200

def verify():
    """Shutdown the Flask server and stop the sweeping thread."""
    global stop_event, sweep_thread, ser

    # Stop the sweeping thread if it's running
    if sweep_thread.is_alive():
        print("Stopping sweeping thread...")
        stop_event.set()  # Signal the sweeping thread to stop
        sweep_thread.join()  # Wait for the thread to finish
        print("Sweeping thread stopped.")
    start_freq = wide_Sweep_Start_Frequency
    end_freq = wide_Sweep_Stop_Frequency
    points = 99  # Number of points in the sweep
    print("\nPerforming Verification Sweep")
    while wideSweepOngoing == True:
        print("Waiting for ongoing wide sweep to complete")
        time.sleep(.1)

    nonCalibratedNetwork = sweep_to_non_calibrated_network(start_freq, end_freq, points)
    calibratedNework = apply_calibration_to_network(nonCalibratedNetwork)


    return calibratedNework

def print_welcome_message():
    message = """#
    ##############################################################
    #                                                            #
    #                   73! de HB9IIU                            #
    #                                                            #
    #   This app is open-source and in continuous development.   #
    #           Thank you for testing and using it!              #
    #                                                            #
    #     For latest news and issues reporting please visit      #
    #                                                            #
    #           https://github.com/HB9IIU/MagnaVNA               #
    #                                                            #
    ##############################################################
    """
    print(message)

def get_vna_port() -> str:
    """
    Automatically detect the NanoVNA serial port by scanning available ports.
    Returns the port if found, raises OSError if the NanoVNA is not detected.
    """
    device_list = list_ports.comports()
    for device in device_list:
        if device.vid == VNA_VID and device.pid == VNA_PID:
            print(f"Found NanoVNA on port: {device.device}")
            return device.device
    raise OSError("NanoVNA device not found")

def initialize_serial(port: str, baudrate=115200, timeout=5):
    """
    Initialize the serial connection with the NanoVNA.
    Args:
    - port: The serial port for the NanoVNA.
    - baudrate: Baud rate for the serial connection (default: 115200).
    - timeout: Timeout in seconds for the serial connection (default: 5).
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"Connected to {port} at {baudrate} baud.")
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None

def send_command(ser, command: str):
    if ser is not None:
        try:
            ser.write((command + '\r\n').encode())
            time.sleep(0.1)
            # print(f"Command sent: {command}")
        except serial.SerialException as e:
            print(f"Error sending command: {e}")
    else:
        print("Serial connection not established.")

def read_response(ser, timeout=15) -> str:
    """
    Read the response from the NanoVNA with a timeout mechanism.
    Args:
    - ser: The serial object.
    - timeout: Time in seconds to wait for the response.
    Returns the filtered response.
    """
    start_time = time.time()
    response = ""

    if ser is not None:
        while True:
            time.sleep(0.1)  # Delay to allow data to accumulate
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                response += data
                if "ch>" in data:  # NanoVNA prompt signals end of response
                    break
            if time.time() - start_time > timeout:  # Timeout condition
                print("Warning: Read timed out")
                break
    else:
        print("Serial connection not established.")

    lines = response.splitlines()
    filtered_lines = [line for line in lines if not (line.startswith(('version', 'scanraw', 'ch>')))]
    return '\n'.join(filtered_lines)

def get_version_info(ser):
    """
    Send the 'version' command to the NanoVNA to retrieve version information.
    Args:
    - ser: The serial object representing the connection to the NanoVNA.
    Returns:
    - The version information as a string.
    """
    send_command(ser, 'version')  # Send 'version' command to the NanoVNA
    return read_response(ser)  # Read and return the response

def set_VNA_calibration_OFF(ser):
    """
    Send the 'version' command to the NanoVNA to retrieve version information.
    Args:
    - ser: The serial object representing the connection to the NanoVNA.
    Returns:
    - The version information as a string.
    """
    send_command(ser, 'cal off')  # Send 'version' command to the NanoVNA
    return read_response(ser)  # Read and return the response

def resetVNA(ser):
    send_command(ser, 'reset')  # Send 'reset' command to the NanoVNA


# --------------------------------------------------
# Core Functions
# --------------------------------------------------

def sweep_to_non_calibrated_network(start_freq, end_freq, points, retries=3, delay=2,network_name="NanoVNA_S11_Non_Calibrated"):

    # Split the sweep into chunks of max 100 points
    max_points = 100
    frequency_chunks = []
    current_start = start_freq
    step_size = int((end_freq - start_freq) / points)

    while current_start < end_freq:
        chunk_stop = min(current_start + step_size * (max_points - 1), end_freq)
        chunk_points = int((chunk_stop - current_start) / step_size) + 1
        frequency_chunks.append((current_start, chunk_stop, chunk_points))
        current_start = chunk_stop + step_size

    print(f"Total Points: {points}, Frequency Chunks: {len(frequency_chunks)}")

    # Collect data from all chunks
    all_frequencies = []
    all_s11_data = []

    start_time = time.time()

    for chunk_start, chunk_stop, chunk_points in frequency_chunks:
        attempt = 0
        while attempt < retries:
            # Send the scan command to the NanoVNA
            command = f"scan {chunk_start} {chunk_stop} {chunk_points} 3\r\n"
            send_command(ser, command)

            # Read the response from the NanoVNA
            response = read_response(ser)
            #print (response)

            # Parse S11 data from the response
            data = []
            frequencies = []
            for line in response.splitlines():
                try:
                    parts = line.split()
                    freq = float(parts[0])
                    real = float(parts[1])
                    imag = float(parts[2])
                    frequencies.append(freq)
                    data.append(complex(real, imag))
                except (ValueError, IndexError):
                    continue

            if data:
                # Append parsed data to the overall results
                all_frequencies.extend(frequencies)
                all_s11_data.extend(data)
                break
            else:
                print(f"Failed to retrieve data for chunk {chunk_start}-{chunk_stop}. Retrying... (Attempt {attempt + 1}/{retries})")
                time.sleep(delay)
                #XXXX
                resetVNA(ser)

                attempt += 1

    # Create the final Network object
    if all_frequencies and all_s11_data:
        frequency_obj = rf.Frequency.from_f(all_frequencies, unit='Hz')
        s_parameters = np.array(all_s11_data).reshape((-1, 1, 1))
        network = rf.Network(frequency=frequency_obj, s=s_parameters, name=network_name)
        end_time = time.time()
        execution_time = round((end_time - start_time),1)
        print ("Sweep results:",network, "Execution time:",execution_time, "[s]")
        return network
    else:
        print("Failed to retrieve data after multiple retries.")
        return None, 0

def apply_calibration_to_network(non_calibrated_network: rf.Network,network_name: str = "NanoVNA_S11_Calibrated") -> rf.Network:
    """
    Apply SOL (Short, Open, Load) calibration to a non-calibrated network object.

    Args:
    - non_calibrated_network: An skrf.Network object containing the raw S11 data (non-calibrated).
    - network_name: Optional. The name for the calibrated network (default: 'Calibrated_Network').

    Returns:
    - The calibrated skrf.Network object.
    """

    # Extract the frequency array from the non-calibrated network
    frequencies = non_calibrated_network.frequency.f  # Frequency array in Hz
    # Save frequencies to a CSV file for debugging (overwriting the file each time)
    # np.savetxt("frequencies_debug.csv", frequencies, delimiter=",")

    # Load measured calibration standards
    measured_short = rf.Network(os.path.join(calibration_kit_dir, 'SHORT_cal.s1p'))
    measured_open = rf.Network(os.path.join(calibration_kit_dir, 'OPEN_cal.s1p'))
    measured_load = rf.Network(os.path.join(calibration_kit_dir, 'LOAD_cal.s1p'))

    # Resample the calibration data to match the frequencies of the non-calibrated network
    measured_short_resampled = measured_short.interpolate(frequencies)
    measured_open_resampled = measured_open.interpolate(frequencies)
    measured_load_resampled = measured_load.interpolate(frequencies)

    # Create ideal Short, Open, and Load standards
    ideal_short = rf.Network(frequency=rf.Frequency.from_f(frequencies, unit='Hz'),
                             s=-1 * np.ones((len(frequencies), 1, 1)))
    ideal_open = rf.Network(frequency=rf.Frequency.from_f(frequencies, unit='Hz'), s=np.ones((len(frequencies), 1, 1)))
    ideal_load = rf.Network(frequency=rf.Frequency.from_f(frequencies, unit='Hz'), s=np.zeros((len(frequencies), 1, 1)))

    # Perform OnePort calibration using measured and ideal standards
    calibration = rf.OnePort(
        ideals=[ideal_short, ideal_open, ideal_load],
        measured=[measured_short_resampled, measured_open_resampled, measured_load_resampled]
    )
    calibration.run()

    # Apply the calibration to the non-calibrated network
    calibrated_dut = calibration.apply_cal(non_calibrated_network)

    # Set the name of the calibrated network
    calibrated_dut.name = network_name

    return calibrated_dut

def save_network_to_file(network: rf.Network, filename: str, folder_name: str = "CalibrationKit"):
    """
    Save an skrf.Network object to a specified folder.
    If the folder does not exist, it is created.
    Args:
    - network: The skrf.Network object to save.
    - filename: The name of the file to save (e.g., 'calibration.s1p').
    - folder_name: The subfolder where the file should be saved (default: 'CalibrationKit').
    """
    folder_path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)
    network.write_touchstone(file_path)
    print(f"Network saved to {file_path}")

def centering_wide_sweep():
    global ws_frequency_array, ws_s11_db_array, ws_swr_array, ws_min_s11_db, ws_min_swr, ws_freq_at_min_s11, ws_freq_at_min_swr
    global wideSweepOngoing
    global recenteringwideSweepOngoing

    # wide sweep
    start_freq = wide_Sweep_Start_Frequency
    end_freq = wide_Sweep_Stop_Frequency
    points =   wide_Sweep_Number_Of_Points
    print("\nPerforming Wide Sweep (Sweep n°" + str(sweepNumber) + ")")
    while narrowSweepOngoing==True:
        print ("Waiting for ongoing narrow sweep to complete")
        time.sleep(.1)
    wideSweepOngoing=True
    nonCalibratedNetwork = sweep_to_non_calibrated_network( start_freq, end_freq, points)
    wideSweepOngoing=False
    recenteringwideSweepOngoing=False
    calibratedNework = apply_calibration_to_network(nonCalibratedNetwork)
    # data for wide sweep plot
    ws_frequency_array = calibratedNework.frequency.f
    ws_s11_db_array = 20 * np.log10(np.abs(calibratedNework.s[:, 0, 0]))
    # Find the first dip (local minima) in S11 and SWR where SWR < swr_threshold
    min_idx = np.argmin(ws_s11_db_array)
    ws_freq_at_min_s11 = ws_frequency_array[min_idx]
    ws_min_s11_db = ws_s11_db_array[min_idx]

def continuous_sweeping_thread():
    global ws_frequency_array, ws_s11_db_array, ws_swr_array, ws_min_s11_db, ws_min_swr, ws_freq_at_min_s11, ws_freq_at_min_swr
    global ns_frequency_array, ns_s11_db_array, ns_swr_array, ns_min_s11_db, ns_min_swr, ns_freq_at_min_s11, ns_freq_at_min_swr
    global sweepNumber
    global latest_zoomed_resonance_frequency
    global narrowSweepOngoing
    global recenteringwideSweepOngoing
    global f1_2, f2_2, f1_3, f2_3, bw_2, bw_3

    """Continuous sweeping thread with stop logic."""
    sweepNumber = 0
    while not stop_event.is_set():  # Check if stop_event is set
        try:
            bw = 300_000
            start_freq = ws_freq_at_min_s11 - bw / 2
            end_freq = ws_freq_at_min_s11 + bw / 2
            points = 400  # Number of points in the sweep
            print("\nPerforming Narrow Sweep (Sweep n°" + str(sweepNumber) + ")")
            while wideSweepOngoing == True:
                print("Waiting for ongoing wide sweep to complete")
                time.sleep(.1)
            narrowSweepOngoing = True
            nonCalibratedNetwork = sweep_to_non_calibrated_network(start_freq, end_freq, points)
            narrowSweepOngoing = False
            calibratedNework = apply_calibration_to_network(nonCalibratedNetwork)
            # Extract the frequency array (in Hz) from the calibrated network object
            # This represents the frequency points of the sweep
            ns_frequency_array = calibratedNework.frequency.f

            # Compute the S11 parameter in decibels (dB)
            # S11 represents the reflection coefficient, and we convert its magnitude to dB
            # Formula: 20 * log10(|S11|), where |S11| is the magnitude of the complex S11 parameter
            ns_s11_db_array = 20 * np.log10(np.abs(calibratedNework.s[:, 0, 0]))

            # Compute the Standing Wave Ratio (SWR) for the same frequency points
            # SWR is derived from the S11 reflection coefficient and is accessed via the skrf property 's_vswr'
            ns_swr_array = calibratedNework.s_vswr[:, 0, 0]

            # --- Step 1: Apply Savitzky-Golay Filtering to Smooth the Data ---
            # Smooth the SWR and S11 data to reduce noise while preserving the shape of the curve
            # window_length: Must be odd and >= the number of data points; polyorder: Degree of the polynomial to fit
            ns_swr_array = savgol_filter(ns_swr_array, window_length=21, polyorder=3)
            ns_s11_db_array = savgol_filter(ns_s11_db_array, window_length=21, polyorder=3)

            # --- Step 2: Slice Arrays to Include Only SWR <= 4 ---
            # Create a boolean mask for SWR values <= 4

            valid_indices = ns_swr_array <= 4.5

            # Slice frequency, SWR, and S11 arrays based on the mask
            # This filters out data points where SWR > 4
            ns_frequency_array = ns_frequency_array[valid_indices]
            ns_swr_array = ns_swr_array[valid_indices]
            ns_s11_db_array = ns_s11_db_array[valid_indices]


            # --- Step 3: Find the Minimum S11 (Local Minima in dB) ---
            # Find the index of the minimum S11 value in the filtered data
            # This identifies the best-matched condition, where the reflection coefficient is minimized
            min_idx = np.argmin(ns_s11_db_array)

            # Retrieve the minimum S11 value (in dB) at the identified index
            ns_min_s11_db = ns_s11_db_array[min_idx]

            # Retrieve the frequency (in Hz) corresponding to the minimum S11 value
            ns_freq_at_min_s11 = ns_frequency_array[min_idx]

            # Retrieve the SWR value at the same index where S11 is minimum
            # This indicates the SWR at the best-matched condition
            ns_min_swr = ns_swr_array[min_idx]

            # Retrieve the frequency (in Hz) corresponding to the minimum SWR value
            # Since the index for minimum S11 and SWR are the same, this value is identical to ns_freq_at_min_s11
            ns_freq_at_min_swr = ns_frequency_array[min_idx]

            # Bandwith calcualtions
            # Define SWR threshold
            swr_threshold = 3  # 3:1 SWR
            # Identify valid indices where SWR <= threshold
            valid_indices = ns_swr_array <= swr_threshold
            # Use valid_indices to extract the filtered frequencies and SWR
            filtered_frequencies = ns_frequency_array[valid_indices]
            filtered_swr = ns_swr_array[valid_indices]
            # f1 and f2 are the first and last frequencies in the filtered range
            f1_3 = filtered_frequencies[0] if len(filtered_frequencies) > 0 else None
            f2_3 = filtered_frequencies[-1] if len(filtered_frequencies) > 0 else None
            # Calculate bandwidth
            bw_3 = f2_3 - f1_3 if f1_3 is not None and f2_3 is not None else 0
            print(f"SWR {swr_threshold}: f1 = {f1_3} Hz, f2 = {f2_3} Hz, BW = {bw_3} Hz")

            # Define SWR threshold
            swr_threshold = 2  # 3:1 SWR
            # Identify valid indices where SWR <= threshold
            valid_indices = ns_swr_array <= swr_threshold
            # Use valid_indices to extract the filtered frequencies and SWR
            filtered_frequencies = ns_frequency_array[valid_indices]
            filtered_swr = ns_swr_array[valid_indices]
            # f1 and f2 are the first and last frequencies in the filtered range
            f1_2 = filtered_frequencies[0] if len(filtered_frequencies) > 0 else None
            f2_2 = filtered_frequencies[-1] if len(filtered_frequencies) > 0 else None
            # Calculate bandwidth
            bw_2 = f2_2 - f1_2 if f1_2 is not None and f2_2 is not None else 0
            print(f"SWR {swr_threshold}: f1 = {f1_2} Hz, f2 = {f2_2} Hz, BW = {bw_2} Hz")

            # Find the updated resonance frequency
            latest_zoomed_resonance_frequency = int(ns_freq_at_min_s11)
            ws_freq_at_min_s11 = latest_zoomed_resonance_frequency
            sweepNumber += 1
            print("Sweep:", sweepNumber, "completed")
        except Exception as e:
            # Log the error with traceback
            message = f"Error at Sweep number {sweepNumber}:\n{str(e)}\n"
            traceback_details = traceback.format_exc()

            print(message)
            if "attempt to get argmin of an empty sequence" in message:
                recenteringwideSweepOngoing=True
                centering_wide_sweep() # recentering
            print("Traceback details:")
            print(traceback_details)

def find_free_port():
    """Find an available port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to a random free port
        return s.getsockname()[1]

def run_flask():
    global port
    #time.sleep(2.5) # to allow serial connection to establish
    port = 5000  # Default port
    try:
        # Check if the port is free
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
    except OSError:
        # If the port is in use, find a random free port
        port = find_free_port()

    print(f"Starting Flask server on port {port}...")


    app.run(host='127.0.0.1', port=port, debug=True, use_reloader=False)

def checkIfCalibrationFilesExist():
    # List of required calibration files
    required_files = ['LOAD_cal.s1p', 'OPEN_cal.s1p', 'SHORT_cal.s1p']
    # Check if all required files exist
    calibrationFilesAvailable = all(
        os.path.exists(os.path.join(calibration_kit_dir, file)) for file in required_files
    )
    print(f"Calibration files available: {calibrationFilesAvailable}")
    return calibrationFilesAvailable
# --------------------------------------------------
# Main Program Flow
# --------------------------------------------------
ser = None  # Initialize ser as None to avoid issues in the exception handler

if __name__ == '__main__':
    print_welcome_message()

    # Check Calibration files
    calibrationFilesAvailable=checkIfCalibrationFilesExist()

    wide_Sweep_Start_Frequency=3_000_000
    wide_Sweep_Stop_Frequency=30_000_000
    wide_Sweep_Number_Of_Points=500

    calibration_start_freq = 1_000_000
    calibration_end_freq = 50_000_000
    calibration_points = 2000  # Number of points in the sweep

    # Suppress Flask default logging
    Flasklog = logging.getLogger('werkzeug')
    Flasklog.setLevel(logging.ERROR)  # Set to ERROR to suppress INFO level logs


    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

# Find VNA, connect, reset, reconnect
    while ser is None:
        try:
            serial_port = get_vna_port()  # Auto-detect NanoVNA port
            ser = initialize_serial(serial_port)  # Initialize the connection
            send_command(ser,"") #empty command to clear buffer
            read_response(ser) #empty buffer
            print("NanoVNA Version Info:", VNA_version_Info)
        except OSError as e:
            print(f"Error: {e}")  # Display error when NanoVNA is not found
            print("NanoVNA device not found. Retrying in 2 seconds.")
            time.sleep(2)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            if ser:  # Close the serial connection only if it was initialized
                ser.close()
                print("Serial connection closed.")
            sys.exit(1)  # Exit with an error code for unexpected exceptions
    resetVNA(ser)
    ser.close()
    ser=None
    time.sleep(1)
    while ser is None:
        try:
            serial_port = get_vna_port()  # Auto-detect NanoVNA port
            ser = initialize_serial(serial_port)  # Initialize the connection
            send_command(ser, "")  # empty command to clear buffer
            read_response(ser)  # empty buffer
            VNA_version_Info = get_version_info(ser)  # Retrieve NanoVNA version
            print("NanoVNA Version Info:", VNA_version_Info)
        except OSError as e:
            print(f"Error: {e}")  # Display error when NanoVNA is not found
            print("NanoVNA device not found. Retrying in 2 seconds.")
            time.sleep(2)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            if ser:  # Close the serial connection only if it was initialized
                ser.close()
                print("Serial connection closed.")
            sys.exit(1)  # Exit with an error code for unexpected exceptions

    set_VNA_calibration_OFF(ser)
    print ("Port:", port)
    # Automatically open the web page in the default browser
    webbrowser.open(f"http://127.0.0.1:{port}")




    # Start the sweep thread if NanoVNA was successfully initialized
    sweep_thread = threading.Thread(target=continuous_sweeping_thread)
    sweep_thread.daemon = True

    if calibrationFilesAvailable:
        centering_wide_sweep()
        sweep_thread.start()
