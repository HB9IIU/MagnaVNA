import shutil
import os
from setuptools import setup

# === Define Folder Paths ===
APP = ['MagnaVNA.py']
DATA_FILES = []

OUTPUT_FOLDER = 'dist'  # Default output folder
BUILD_FOLDER = 'build'  # Temporary build folder
FINAL_FOLDER = 'MacApp'  # Where the final ZIP will be stored
APP_NAME = 'MagnaVNA.app'
ZIP_NAME = 'MagnaVNA_MacOS.zip'

# === Cleanup Old Folders Before Building ===
def safe_remove(folder):
    """Safely removes a folder if it exists."""
    if os.path.exists(folder):
        print(f"🔄 Removing old {folder} folder...")
        shutil.rmtree(folder, ignore_errors=True)
        print(f"✅ {folder} removed.")

safe_remove(OUTPUT_FOLDER)
safe_remove(BUILD_FOLDER)
safe_remove(FINAL_FOLDER)

# === Setup Options ===
OPTIONS = {
    'argv_emulation': True,
    'iconfile': 'static/macIcon/magnavna.icns',
    'includes': [
        'serial', 'time', 'numpy', 'skrf', 'typing_extensions', 'sys',
        'serial.tools.list_ports', 'os', 'threading', 'flask', 'flask.json',
        'flask.render_template', 'logging', 'scipy.signal.savgol_filter',
        'webbrowser', 'traceback', 'socket'
    ],
    'packages': ['flask', 'skrf', 'scipy'],
    'resources': ['static', 'templates', 'CalibrationKit'],
    'excludes': [
        '.DS_Store', 'PySide6', 'matplotlib', 'PyQt5', 'gi.repository', 'win32wnet', 'winreg', 'test'
    ],
}

# === Run the Build Process ===
setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)

# === Post-Build: Zip and Move the App ===
app_path = os.path.join(OUTPUT_FOLDER, APP_NAME)
zip_path = os.path.join(OUTPUT_FOLDER, ZIP_NAME)
final_zip_path = os.path.join(FINAL_FOLDER, ZIP_NAME)

if os.path.exists(app_path):
    print(f"📦 Zipping {APP_NAME} into {ZIP_NAME}...")
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', OUTPUT_FOLDER, APP_NAME)
    print(f"✅ Created {ZIP_NAME} in {OUTPUT_FOLDER}")

    # === Ensure MacApp folder exists ===
    os.makedirs(FINAL_FOLDER, exist_ok=True)

    # === Move ZIP to MacApp ===
    shutil.move(zip_path, final_zip_path)
    print(f"📂 Moved {ZIP_NAME} to {FINAL_FOLDER}/")

    # === Cleanup the dist folder ===
    safe_remove(OUTPUT_FOLDER)

# === Cleanup the build folder ===
safe_remove(BUILD_FOLDER)

print(f"🎉 Done! Find your packaged app in {FINAL_FOLDER}/{ZIP_NAME}")
