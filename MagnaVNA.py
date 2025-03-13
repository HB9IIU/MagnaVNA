
import os
import sys
import skrf as rf  # scikit-rf  Requires typing_extensions
import serial
import time
import numpy as np
from serial.tools import list_ports
import threading
from flask import Flask, request, jsonify, render_template
from scipy.signal import savgol_filter
import webbrowser
import traceback
import socket
global port
import platform
import shutil
from werkzeug.serving import make_server
import subprocess
import logging
##################################################################
VERSION="1.2" # 15th March 2025

wide_Sweep_Start_Frequency = 3_000_000
wide_Sweep_Stop_Frequency = 30_000_000
wide_Sweep_Number_Of_Points = 500  # Number of points in the sweep

calibration_start_freq = 1_000_000
calibration_end_freq = 50_000_000
calibration_points = 2000  # Number of points in the sweep

###################################################################



# Global Event to signal thread stop
stop_event = threading.Event()

firstCenteringSweepCompleted=False #to wait before we go to narrow sweep

skipFirstWideSweepRequestFromWebPage=True

flask_ready = False  # Global flag to indicate when Flask is fully running

global ws_frequency_array, ws_s11_db_array, ws_swr_array, ws_min_s11_db, ws_min_swr, ws_freq_at_min_s11, ws_freq_at_min_swr
global ns_frequency_array, ns_s11_db_array, ns_swr_array, ns_min_s11_db, ns_min_swr, ns_freq_at_min_s11, ns_freq_at_min_swr
global parallel_R, parallel_X, parallel_L, resonance_impedance, s11_phase_resonance, series_L, series_C, quality_factor, reactance_type
global NarrowSweepNumber, latest_zoomed_resonance_frequency, narrowSweepOngoing, recenteringWideSweepOngoing
global f1_2, f2_2, f1_3, f2_3, bw_2, bw_3
global disconnectionAlarm
wideSweepOngoing=False
narrowSweepOngoing=False
recenteringWideSweepOngoing=False
calibrationFilesAvailable=False
disconnectionAlarm=False

VERSION= "Version " + VERSION





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





# Get the current system's OS
print("Possible matplolib error message can be ignored.....")
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

def copyWSPRconfigFile_NOT_IN_USE():
    # NEW copy wspr config file
    config_file = os.path.join(BASE_DIR, 'wsprConfig.cfg')
    destination_file = os.path.join(template_folder, 'wsprConfig.js')
    if os.path.exists(config_file):
        shutil.copy2(config_file, destination_file)  # Overwrites if exists
        #print(f"Copied '{config_file}' to '{destination_file}' (overwriting if necessary).")
        print( "[INFO] WSPR config file successfully overwritten.")
    else:
        #print(f"File '{config_file}' not found, skipping copy.")
        print ("[WARNING] WSPR Config file not found, skipping copy.")


# --------------------------------------------------
# Flask Routes
# --------------------------------------------------
# Flask app initialization with explicit paths
app = Flask(
    __name__,
    template_folder=template_folder,  # Absolute path to templates
    static_folder=static_folder       # Absolute path to static files
)



#------------------------------------------------------------------------
# ENDPOINTS
#------------------------------------------------------------------------

@app.route('/')
def index_default():
    return render_template('index.html')

@app.route('/theory.html')
def theory():
    return render_template('theory.html')

@app.route('/HB9IIUwsprReporter.html')
def HB9IIUwsprReporter():
    return render_template('HB9IIUwsprReporter.html')

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
        'AppVersion': VERSION,
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
    global disconnectionAlarm
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
        'recenteringWideSweepOngoing': recenteringWideSweepOngoing,
        'parallel_R':parallel_R,
        'parallel_X': parallel_X,
        'parallel_L': parallel_L,
        'resonance_impedance': resonance_impedance,
        's11_phase_resonance': s11_phase_resonance,
        'series_L': series_L,
        'series_C': series_C,
        'quality_factor': quality_factor,
        'reactance_type': reactance_type,
        'disconnectionAlarm': disconnectionAlarm
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
    global skipFirstWideSweepRequestFromWebPage
    if (skipFirstWideSweepRequestFromWebPage or wideSweepOngoing):
        skipFirstWideSweepRequestFromWebPage=False
        return jsonify({'message': 'Skipping 1st Wide Sweep Request!'})
    else:
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


#------------------------------------------------------------------------
# FUNCTIONS
#------------------------------------------------------------------------
# --------------------------------------------------
# GENERATE DUMMY CALIBRATION FILES
# --------------------------------------------------
def generate_dummy_calibration_files(start_freq, end_freq, num_points):
    """Generate dummy calibration files (SHORT, OPEN, LOAD) that have no impact on calibration."""
    frequencies = np.linspace(start_freq, end_freq, num_points)

    # Generate ideal S-parameters
    s11_short = -1 * np.ones((num_points, 1, 1))  # SHORT = -1
    s11_open = 1 * np.ones((num_points, 1, 1))  # OPEN = +1
    s11_load = 0 * np.ones((num_points, 1, 1))  # LOAD = 0

    # Create scikit-rf Network objects
    freq_obj = rf.Frequency.from_f(frequencies, unit='Hz')

    short_network = rf.Network(frequency=freq_obj, s=s11_short, name="SHORT")
    open_network = rf.Network(frequency=freq_obj, s=s11_open, name="OPEN")
    load_network = rf.Network(frequency=freq_obj, s=s11_load, name="LOAD")

    # Save to files
    short_network.write_touchstone(os.path.join(calibration_kit_dir, 'SHORT_cal'))
    open_network.write_touchstone(os.path.join(calibration_kit_dir, 'OPEN_cal'))
    load_network.write_touchstone(os.path.join(calibration_kit_dir, 'LOAD_cal'))

    print("[INFO] Dummy calibration files created in CalibrationKit folder.")


# Example usage:
# generate_dummy_calibration_files(1_000_000, 50_000_000, 2000)

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
    calibratedNetwork = apply_calibration_to_network(nonCalibratedNetwork)


    return calibratedNetwork

def print_welcome_message():
    message = f"""
    ##############################################################
    #                      73! de HB9IIU                         #
    #   MagnaVNA is open-source and in continuous development    #
    #           Thank you for testing and using it!              #
    #     For latest news and issues reporting please visit      #
    #                     VERSION {VERSION}                    #
    #            https://github.com/HB9IIU/MagnaVNA              #
    ##############################################################
    """

    print(message)

# Get the local IP address (IPv4) of the machine
def get_local_ip():
    try:
        # Create a temporary socket and connect to an external server (doesn't actually send data)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # Google's public DNS
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"  # Fallback

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
    #measured_short = rf.Network(os.path.join(calibration_kit_dir, 'SHORT_cal.s1p'))
    #measured_open = rf.Network(os.path.join(calibration_kit_dir, 'OPEN_cal.s1p'))
    #measured_load = rf.Network(os.path.join(calibration_kit_dir, 'LOAD_cal.s1p'))

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
    global recenteringWideSweepOngoing
    global firstCenteringSweepCompleted

    # wide sweep
    start_freq = wide_Sweep_Start_Frequency
    end_freq = wide_Sweep_Stop_Frequency
    points =   wide_Sweep_Number_Of_Points
    print("\nPerforming Wide Sweep.................")
    while narrowSweepOngoing==True:
        print ("Waiting for ongoing narrow sweep to complete")
        time.sleep(.1)
    wideSweepOngoing=True
    nonCalibratedNetwork = sweep_to_non_calibrated_network( start_freq, end_freq, points)
    wideSweepOngoing=False
    recenteringWideSweepOngoing=False
    calibratedNetwork = apply_calibration_to_network(nonCalibratedNetwork)
    #calibratedNetwork=nonCalibratedNetwork
    resonance_idx = np.argmin(calibratedNetwork.s_mag[:, 0, 0])  # Index of min S11 magnitude
    ws_freq_at_min_s11 = calibratedNetwork.frequency.f[resonance_idx]  # Resonant frequency
    ws_frequency_array = calibratedNetwork.frequency.f
    ws_s11_db_array = 20 * np.log10(np.abs(calibratedNetwork.s[:, 0, 0]))
    print(f"Resonance Frequency according to Wide Sweep: {ws_freq_at_min_s11:_}")
    firstCenteringSweepCompleted = True

def continuous_sweeping_thread():
    """
    Continuous sweeping function that performs narrow sweeps, extracts key RF parameters,
    applies smoothing, and updates resonance information.
    """

    # Global variables for Flask access
    global ws_frequency_array, ws_s11_db_array, ws_swr_array, ws_min_s11_db, ws_min_swr, ws_freq_at_min_s11, ws_freq_at_min_swr
    global ns_frequency_array, ns_s11_db_array, ns_swr_array, ns_min_s11_db, ns_min_swr, ns_freq_at_min_s11, ns_freq_at_min_swr
    global parallel_R, parallel_X, parallel_L, resonance_impedance, s11_phase_resonance, series_L, series_C,quality_factor
    global NarrowSweepNumber, latest_zoomed_resonance_frequency, narrowSweepOngoing, recenteringWideSweepOngoing, reactance_type
    global f1_2, f2_2, f1_3, f2_3, bw_2, bw_3

    # Wait if a wide sweep is ongoing
    while firstCenteringSweepCompleted==False:
        print("Waiting for very first wide sweep to complete" )
        time.sleep(0.5)


    NarrowSweepNumber = 0  # Initialize sweep counter
    #initial Sweep bw
    FirstSweepBandwidth =1_000_000  # Sweep bandwidth in Hz
    while not stop_event.is_set():  # Continuous loop unless stopped
        try:
            # Define sweep parameters
            if NarrowSweepNumber==0:
                start_freq = ws_freq_at_min_s11 - FirstSweepBandwidth / 2
                end_freq = ws_freq_at_min_s11 + FirstSweepBandwidth / 2
                points = 600  # Number of points in the sweep
            else:
                start_freq = resonance_freq - sweepBandwidth / 2
                end_freq = resonance_freq + sweepBandwidth / 2
                points = 400  # Number of points in the sweep

            print(f"\nPerforming Narrow Sweep (Sweep n°{NarrowSweepNumber})")

            # Wait if a wide sweep is ongoing
            while wideSweepOngoing:
                print("Waiting for ongoing wide sweep to complete...")
                time.sleep(1)

            # Start narrow sweep
            narrowSweepOngoing = True # used for Flask not erturning data
            nonCalibratedNetwork = sweep_to_non_calibrated_network(start_freq, end_freq, points)
            narrowSweepOngoing = False
            # Apply calibration to measurement
            calibratedNetwork = apply_calibration_to_network(nonCalibratedNetwork)
            # Extract frequency array (Hz) and impedance (Z = R + jX)
            ns_frequency_array = calibratedNetwork.frequency.f
            Z = calibratedNetwork.z[:, 0, 0]  # Complex impedance array
            R, X = Z.real, Z.imag  # Extract real and imaginary parts

            # Identify the resonance point (minimum S11)
            min_idx = np.argmin(calibratedNetwork.s_mag[:, 0, 0])
            resonance_freq = int(ns_frequency_array[min_idx] ) # Frequency at resonance (Hz)

            resonance_R = R[min_idx]  # Resistance at resonance
            resonance_X = X[min_idx]  # Reactance at resonance
            reactance_type = "Inductive" if resonance_X > 0 else "Capacitive"

            # Format impedance string at resonance
            resonance_impedance = f"{resonance_R:.1f} {'+' if resonance_X >= 0 else '-'} j {abs(resonance_X):.1f} Ω"

            # Compute S11 phase in degrees at resonance
            s11_complex = calibratedNetwork.s[:, 0, 0]  # Extract S11 as complex numbers
            s11_phase_array = np.angle(s11_complex, deg=True)
            s11_phase_resonance = s11_phase_array[min_idx]

            # Compute Series Inductance (L) or Capacitance (C)
            if resonance_X > 0:  # Inductive reactance
                series_L = (resonance_X / (2 * np.pi * resonance_freq)) * 1e9  # Convert to nH
                series_C = None
            else:  # Capacitive reactance
                series_C = (1 / (2 * np.pi * resonance_freq * abs(resonance_X))) * 1e9  # Convert to nF
                series_L = None

            # Compute Parallel Resistance (Rp) and Parallel Reactance (Xp)
            parallel_R = resonance_R * (1 + (resonance_X / resonance_R) ** 2)
            parallel_X = (resonance_X * resonance_R ** 2) / (resonance_R ** 2 + resonance_X ** 2)
            parallel_L = (parallel_X / (2 * np.pi * resonance_freq)) * 1e6  # Convert to µH

            # Compute Return Loss (RL) in dB
            return_loss = -20 * np.log10(np.abs(calibratedNetwork.s[min_idx, 0, 0]))
            '''
            Print extracted parameters
            print(f"Return Loss at resonance: {return_loss:.2f} dB")
            print(f"Impedance at resonance: {resonance_impedance}")
            print(f"S11 Phase at resonance: {s11_phase_resonance:.2f}°")

            if series_L is not None:
                print(f"Series L: {series_L:.3f} nH")
            elif series_C is not None:
                print(f"Series C: {series_C:.4f} nF")

            print(f"Parallel R: {parallel_R:.3f} Ω")
            print(f"Parallel X: {parallel_L:.4f} µH")  # Converted from Xp
            '''
            # Apply Savitzky-Golay filter for smoothing SWR and S11
            window_length = min(21, len(ns_frequency_array) - 1)  # Ensure valid window size
            window_length = window_length - 1 if window_length % 2 == 0 else window_length  # Ensure it's odd
            ns_s11_db_array = savgol_filter(20 * np.log10(np.abs(calibratedNetwork.s[:, 0, 0])), window_length, polyorder=3)
            ns_swr_array = savgol_filter(calibratedNetwork.s_vswr[:, 0, 0], window_length, polyorder=3)
            ns_min_swr = np.min(ns_swr_array)

            # Get the index of the minimum SWR
            min_swr_index = np.argmin(ns_swr_array)
            #print(f"SWR: {ns_min_swr}")

            #print(f"Index of minimum SWR: {min_swr_index}")
            ns_freq_at_min_s11 = int(ns_frequency_array[min_swr_index] ) # Frequency at resonance (Hz)
            print(f"Resonance Frequency from min returned S11: {resonance_freq:,d} Hz")
            print(f"Resonance Frequency from min smoothed S11: {ns_freq_at_min_s11:,d} Hz")

            # Bandwidth calculations at SWR thresholds 3:1 and 2:1
            def compute_bandwidth(swr_threshold):
                indices = ns_swr_array <= swr_threshold
                if np.any(indices):
                    f_low, f_high = ns_frequency_array[indices][0], ns_frequency_array[indices][-1]
                    return f_low, f_high, f_high - f_low
                return None, None, 0
            f1_3, f2_3, bw_3 = compute_bandwidth(3)  # SWR 3:1
            f1_2, f2_2, bw_2 = compute_bandwidth(2)  # SWR 2:1
            f1_forSweepBW, f2_forSweepBW, bw_forSweepBW = compute_bandwidth(4.5) # SWR 4.5:1  will be used for next sweep definition

            # Check if all the values are valid

            if f1_3 is not None and f2_3 is not None and bw_3 > 0:
                print(f"SWR 3:1 -> BW = {bw_3 / 1e3:.2f} kHz, Range: {f1_3 / 1e6:.3f} - {f2_3 / 1e6:.3f} MHz")
            else:
                print("No valid frequency range found for SWR 3:1")

            if f1_2 is not None and f2_2 is not None and bw_2 > 0:
                print(f"SWR 2:1 -> BW = {bw_2 / 1e3:.2f} kHz, Range: {f1_2 / 1e6:.3f} - {f2_2 / 1e6:.3f} MHz")
            else:
                print("No valid frequency range found for SWR 2:1")

            if bw_forSweepBW > 0:
                print(f"SWR 5:1 -> BW = {bw_forSweepBW / 1e3:.2f} kHz, Range: {f1_forSweepBW / 1e6:.3f} - {f2_forSweepBW / 1e6:.3f} MHz")
            else:
                print("No valid frequency range found for SWR 4.5:1")
                recenteringWideSweepOngoing = True
                centering_wide_sweep()
                resonance_freq=ws_freq_at_min_s11
                sweepBandwidth = FirstSweepBandwidth
                continue


            # Compute Quality Factor using bandwidth at SWR 2:1
            quality_factor = resonance_freq / bw_2 if bw_2 > 0 else None
            #if quality_factor is not None:
                #print(f"Quality Factor (Q): {quality_factor:.3f}")


            # Round values for Flask display
            parallel_R = round(parallel_R, 2)
            parallel_X = round(parallel_X, 2)
            parallel_L = round(parallel_L, 4)
            s11_phase_resonance = round(s11_phase_resonance, 1)
            series_L = round(series_L, 3) if series_L is not None else None
            series_C = round(series_C, 4) if series_C is not None else None
            ns_min_swr= round(ns_min_swr, 1)
            quality_factor=round(quality_factor, 0)

            # data for used by Flask
            '''
            print("\n--- Values returned by Flask ---")
            print(f"Parallel R (parallel_R): {parallel_R} Ω")
            print(f"Parallel X (parallel_X): {parallel_X} Ω")
            print(f"Parallel L (parallel_L): {parallel_L} µH")
            print(f"Impedance at Resonance (resonance_impedance): {resonance_impedance}")
            print(f"S11 Phase at Resonance (s11_phase_resonance): {s11_phase_resonance}°")
            print(f"Series L (series_L): {series_L} nH" if series_L is not None else "Series L: N/A")
            print(f"Series C (series_C): {series_C} nF" if series_C is not None else "Series C: N/A")
            print(f"SWR (ns_min_swr): {ns_min_swr}")
            '''

            # Use the already computed minimum SWR index
            resonance_index = min_swr_index  # Index of minimum SWR
            resonance_freq = ns_freq_at_min_s11  # Resonance frequency (Hz), already extracted

            # Adjustable thresholds
            FINAL_SWR_THRESHOLD = 4.0  # Final zoom-in threshold

            # Step 1: Get Total Points Before Any Slicing
            NumberOfPointsBeforeFiltering = len(ns_swr_array)
            lowest_swr_index = np.argmin(ns_swr_array)  # Index of lowest SWR value

            # 🔹 Print initial data
            print(f"Before Broad Slice - SWR: {ns_swr_array[0]:.1f}  <--> {ns_swr_array[-1]:.1f}")
            print(f"Total Points Before Slicing: {NumberOfPointsBeforeFiltering}")
            print(
                f"Lowest SWR before slicing: {ns_swr_array[lowest_swr_index]:.1f} at index {lowest_swr_index}")

            # **Step 1: Centering**
            half_size = NumberOfPointsBeforeFiltering // 2
            start_index = max(lowest_swr_index - half_size, 0)
            end_index = min(lowest_swr_index + half_size, NumberOfPointsBeforeFiltering - 1)

            # Slice the arrays to center around the lowest SWR
            ns_frequency_array = ns_frequency_array[start_index:end_index + 1]
            ns_swr_array = ns_swr_array[start_index:end_index + 1]
            ns_s11_db_array = ns_s11_db_array[start_index:end_index + 1]

            # Compute new size
            new_lowest_swr_index = np.argmin(ns_swr_array)
            points_left = new_lowest_swr_index
            points_right = len(ns_swr_array) - new_lowest_swr_index - 1

            # 🔹 Print after centering
            print(f"After Centering - SWR: {ns_swr_array[0]:.1f}  <--> {ns_swr_array[-1]:.1f}")
            print(f"Centered Array Points: {points_left} <-|-> {points_right}")

            # **Step 2: Zooming (Trimming symmetrically)**
            while points_left > 0 and points_right > 0:
                # Stop trimming if either edge has reached SWR 4.2
                if ns_swr_array[0] <= FINAL_SWR_THRESHOLD or ns_swr_array[-1] <= FINAL_SWR_THRESHOLD:
                    break

                # Remove one point from each side
                ns_frequency_array = ns_frequency_array[1:-1]
                ns_swr_array = ns_swr_array[1:-1]
                ns_s11_db_array = ns_s11_db_array[1:-1]

                # Update new size
                new_lowest_swr_index = np.argmin(ns_swr_array)
                points_left = new_lowest_swr_index
                points_right = len(ns_swr_array) - new_lowest_swr_index - 1

            # **Final Total Points After Slicing**
            NumberOfPointsAfterFiltering = len(ns_swr_array)

            # 🔹 Print final results
            print(f"Final Slice (SWR {FINAL_SWR_THRESHOLD}) - SWR: {ns_swr_array[0]:.1f}  <--> {ns_swr_array[-1]:.1f}")
            print(f"Total Points After Slicing: {NumberOfPointsAfterFiltering}")
            print(f"Left & Right points: {points_left} <-|-> {points_right}")
            sweepBandwidth=bw_forSweepBW*1.2
            NarrowSweepNumber += 1
            print ()
            print(f"Sweep {NarrowSweepNumber} completed")
            #time.sleep(5)  for debugging
        except Exception as e:
            print(f"Error durin Sweep number {NarrowSweepNumber}: {e}")
            print(traceback.format_exc())

def run_flask():
    global port, flask_ready
    port = 5556  # Default port
    local_ip = get_local_ip()

    def get_free_port():
        """Find a free port dynamically."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))  # Bind to any free port
            return s.getsockname()[1]

    def is_port_in_use(port):
        """Cross-platform method to check if a port is in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return False
            except OSError:
                return True

    # Step 1: Try port 5555, otherwise find a free port
    if is_port_in_use(port):
        print(f"[WARNING] Port {port} is busy. Finding a free port...")
        port = get_free_port()
        print(f"[INFO] Switched to available port {port}.")
    else:
        print(f"[INFO] Port {port} is free. Using it for Flask.")

    print(f"[INFO] Flask will start on http://{local_ip}:{port}")

    # Step 2: Define a threaded Flask server
    class FlaskThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            print("[INFO] Initializing Flask server...")
            self.server = make_server(local_ip, port, app)
            self.ctx = app.app_context()
            self.ctx.push()
            print("[INFO] Flask server initialized successfully.")

        def run(self):
            global flask_ready
            print(f"[INFO] Flask is now running on http://{local_ip}:{port}")
            flask_ready = True  # 🔹 Set the flag when Flask is fully running
            self.server.serve_forever()

        def shutdown(self):
            print("[INFO] Shutting down Flask server...")
            self.server.shutdown()
            print("[INFO] Flask server has been shut down.")

    # Step 3: Start Flask in a separate thread
    print("[INFO] Starting Flask server thread...")
    flask_thread = FlaskThread()
    flask_thread.start()

    # Step 4: Open the browser only when Flask is ready
    while not flask_ready:
        print("\n[INFO] Waiting for Flask to be ready...")
        time.sleep(0.5)  # Small wait instead of fixed sleep

    print(f"[INFO] Opening Flask web interface in browser: http://{local_ip}:{port}")
    webbrowser.open(f"http://{local_ip}:{port}")

def checkIfCalibrationFilesExist():
    # List of required calibration files
    required_files = ['LOAD_cal.s1p', 'OPEN_cal.s1p', 'SHORT_cal.s1p']
    # Check if all required files exist
    calibrationFilesAvailable = all(
        os.path.exists(os.path.join(calibration_kit_dir, file)) for file in required_files
    )
    return calibrationFilesAvailable


# NEW SERIAL

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
    global ser, VNA_version_Info,disconnectionAlarm
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
                        disconnectionAlarm=True
                        print("❌ [ERROR] No response from VNA, unplug & replug")
                print("[INFO] VNA reset")
                command = "reset\r"
                ser.write(command.encode('utf-8'))
                ser.close()
                time.sleep(3)
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
                    print("AX raw for debug only:", repr(response))
                    time.sleep(0.2)

                # Extract version info using split
                lines = response.split('\r\n')
                VNA_version_Info = lines[1] if len(lines) > 1 else None
                # print(f"Received response: {response}")
                disconnectionAlarm = False

                print(f"[INFO] VNA version info: {VNA_version_Info}")
                print(f"✅[INFO] VNA now Ready to Receive Commands")
                return
            else:
                print("[ERROR] Failed to open serial connection.")
        except Exception as e:
            print(f"[ERROR] {e}")
        print(f"[WARNING] Retrying connection in {retry_delay} seconds...")
        disconnectionAlarm = True
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
    start_freq=int(start_freq)
    end_freq=int(end_freq)
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
        time.sleep(0)
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




# --------------------------------------------------
# Main Program Flow
# --------------------------------------------------



if __name__ == '__main__':
    print_welcome_message()

    # Example usage:
    # generate_dummy_calibration_files(1_000_000, 50_000_000, 2000)
    #sys.exit(0)

    # Check Calibration files
    calibrationFilesAvailable = checkIfCalibrationFilesExist()
    print(f"[INFO] Calibration files available: {calibrationFilesAvailable}\n")

    # Suppress Flask logging
    Flasklog = logging.getLogger('werkzeug')
    Flasklog.setLevel(logging.ERROR)

    # Start Flask in a separate thread
    print("[INFO] Starting Flask web server...")
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    VNAconnectOrReconnect()




    # 🔹 Wait until Flask is ready before proceeding
    while not flask_ready:
        print("[INFO] Waiting for Flask to be fully ready...")
        time.sleep(0.5)  # Short wait loop
    time.sleep(1) # ensure web page has loaded
    print("[INFO] Flask initialization complete. Proceeding with NanoVNA setup...\n")



    measured_short = rf.Network(os.path.join(calibration_kit_dir, 'SHORT_cal.s1p'))
    measured_open = rf.Network(os.path.join(calibration_kit_dir, 'OPEN_cal.s1p'))
    measured_load = rf.Network(os.path.join(calibration_kit_dir, 'LOAD_cal.s1p'))




    centering_wide_sweep()

    # --- Start Sweep Thread ---
    sweep_thread = threading.Thread(target=continuous_sweeping_thread)
    sweep_thread.daemon = True

    if calibrationFilesAvailable:
        print("[INFO] Calibration is available. Continuozs Sweeping started.\n")
        sweep_thread.start()

''' WORK IN PROGRESS

# Sample config data
config = {
    "callsign": "HB9IIU",
    "latitude": "47.3667",
    "longitude": "8.5500",
    "cesium_api_key": ""
}

@app.route('/get_config', methods=['GET'])
def get_config():
    return jsonify(config)

@app.route('/save_config', methods=['POST'])
def save_config():
    data = request.json
    config['callsign'] = data.get('callsign', config['callsign'])
    config['latitude'] = data.get('latitude', config['latitude'])
    config['longitude'] = data.get('longitude', config['longitude'])
    config['cesium_api_key'] = data.get('cesium_api_key', config['cesium_api_key'])
    return jsonify({"status": "success"})





'''















