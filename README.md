# MagnaVNA for Windows

This is the very first commit of the MagnaVNA app for Windows. It’s a packaged version of a Python application that I will share later (once I’ve cleaned up the code).

## What is MagnaVNA?

MagnaVNA is a simple tool designed for users like me, who have struggled with continuously readjusting stimulus values through the NanoVNA touch screen while manually tuning a magnetic loop during initial trials. This app streamlines the process and makes tuning much easier by automating certain steps.

## What it does

The app first performs a sweep from 1 to 30 MHz to identify the resonant frequency of your antenna. After that, it continuously sweeps around this point, automatically readjusting while you’re adjusting the capacitor of your antenna. This allows you to focus on the physical tuning without constantly needing to tweak the settings.

## Demo

Here is a demo video showing how MagnaVNA works:  
[![Watch the demo on YouTube](https://img.youtube.com/vi/amZQqX4hNA8/0.jpg)](https://youtu.be/amZQqX4hNA8)

## Note

Please don’t be discouraged if it doesn’t work perfectly the first time. Remember, this is a hobby, and we are just experimenting together!

This app works very well for me using the [ESP32-MLA-Manual-Tuner](https://github.com/HB9IIU/ESP32-MLA-Manual-Tuner).

I’m sorry if it takes some time to respond to all issues. I’m also busy with other projects, but I appreciate your understanding and patience.

## Reporting Issues or Feedback

If you encounter any bugs, issues, or have suggestions for new features, please open an issue on the **Issues** tab of this repository. When reporting a bug, please try to provide the following details:
- Description of the issue
- Steps to reproduce (if applicable)
- Any error messages you see

You can also open an issue for general feedback or improvement suggestions.

Thank you for your help and feedback as we improve this tool!

73! de HB9IIU
