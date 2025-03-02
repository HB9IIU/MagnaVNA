# MagnaVNA

## What is MagnaVNA?

MagnaVNA is a simple tool designed to be used with one of the inexpensive Chinese VNAs from the NanoVNA family.

It was created for users like me with big fingers and bad eyes who sometimes forget how to navigate the menus and struggle with continuously readjusting stimulus values and other parameters through the tiny touch screen while manually tuning a magnetic loop antenna during initial trials.

This app streamlines the process and makes tuning much easier by automating certain steps.

## What it does

The app first asks the VNA to perform a sweep from 1 to 30 MHz and collects raw data to identify the resonant frequency of your antenna. It then displays key measurements:

- **S11 (Reflection Coefficient):** This tells you how much of the signal is reflected back from the antenna instead of being transmitted. A lower S11 value means better efficiency.
- **SWR (Standing Wave Ratio):** This indicates how well the antenna is matched to the transmission line. A lower SWR means less power is wasted and more is radiated.
- **Resonance Frequency:** The frequency at which S11 is at its lowest, indicating the best match.
- **Bandwidth at SWR 2:** The frequency range where the SWR remains below 2, showing the effective tuning range.
- **Q Factor:** A measure of how sharp the resonance is, calculated from the bandwidth and resonance frequency.

These values are shown in a simple web interface in the form of well-known curves, similar to those on the NanoVNA screen but displayed much larger for better visibility and easier analysis.

After the initial sweep, the app continuously measures and updates around the resonance point while you adjust the capacitor of your antenna.

This allows you to focus on the physical tuning without constantly needing to tweak the settings.

## Important Note

The package includes calibration files that are mine, therefore, they obviously do not fit your setup. Although you might see an acceptable output, it is **strongly recommended** to carry out calibration via the relevant page. Click **Re-Calibrate** at the bottom of the screen to perform a proper calibration. You can then verify your calibration using the **Check Calibration** button.

## How to Use MagnaVNA

### Windows

For Windows users, I initially attempted to create a standalone **.exe** file from the Python script using tools like **PyInstaller** and other similar solutions. However, every attempt resulted in false virus warnings, making it difficult for users to install and run the software without security concerns.

To avoid this issue, I decided to use **WinPython**, which provides a portable Python environment. This means you do not need to install Python separately—everything needed to run MagnaVNA is included in the package.

#### Steps to Use:
1. **Download** the provided package.
2. **Extract** the contents to a folder of your choice.
3. **Run MagnaVNA** by double-clicking either:
   - `MagnaVNA.bat` (recommended for ensuring all dependencies load correctly) or
   - `MagnaVNA.exe` (for direct execution).

No installation is required; just extract and run!

If anyone is experienced with **creating a standalone EXE** that does not trigger false virus warnings, I would be extremely grateful for assistance.

### macOS

For macOS users, I used **py2app** to package the application into a macOS-compatible format. Since macOS displays the full package contents if not zipped, the app is provided as a **zip file**.

#### Steps to Use:
1. **Download** the provided package.
2. **Unzip** `MagnaVNA.zip`.
3. **Run the app** by opening the extracted `.app` file.

The macOS version is designed to run smoothly without requiring additional dependencies. Let me know if you run into any issues!

## Tested Devices

The app should run smoothly with the **classic entry-level NanoVNA** as well as with the **larger-screen NanoVNA-H4**.

I am working on a new version that will also support **NanoVNA-V2 and LiteVNA 64**, but I am not there yet.

## Reporting Issues, Bugs, and Suggestions

MagnaVNA is just a hobby project, and I’m working on it in my free time. If you have any questions, encounter bugs, or have suggestions for improvements, please create an **issue** in the repository.

When opening an issue, it helps if you can include:
- A description of the issue
- Steps to reproduce (if applicable)
- Any error messages you see

I appreciate your patience in waiting for my responses—I’ll do my best to reply as soon as I can. Thank you for your understanding and support!

---

This guide is meant to make MagnaVNA as easy as possible to use, regardless of your platform. Let me know if you have any issues or suggestions!
