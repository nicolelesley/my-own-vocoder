# MidiCam Python

A Python-based alternative to MidiCam that captures hand and facial gestures via webcam and converts them to MIDI signals for vocal production in Ableton Live, Cubase, or other DAWs.

![MidiCam Python](https://i.imgur.com/VkjMX0w.jpg)

## Features

- Hand gesture detection for controlling various vocal parameters
- Facial gesture recognition for additional control
- Real-time MIDI control change messages
- Visual feedback of detected gestures and parameter values
- Configurable MIDI port and controller mappings
- Modern UI with multiple color themes
- Preset system activated by specific hand gestures
- Anti-glitch filtering for smooth parameter changes

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- Mido (MIDI library)
- NumPy
- Virtual MIDI device (IAC Driver on Mac, loopMIDI on Windows)

## Installation

1. Install the required Python packages:

```bash
pip install -r requirements.txt
# or
pip install opencv-python mediapipe mido numpy python-rtmidi
```

2. Set up a virtual MIDI device:
   - **Mac**: Enable the IAC Driver in Audio MIDI Setup
   - **Windows**: Install and configure [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html)
     - Create at least one virtual MIDI port (e.g., "loopMIDI Port 1")

3. Configure your DAW (Ableton Live, Cubase, etc.) to receive MIDI from the virtual MIDI port

## MIDI Setup

1. Run the MIDI setup utility to configure your MIDI connection:

```bash
python midi_setup.py
```

This will:
- List available MIDI ports on your system
- Let you select a port to use
- Test the connection with sample MIDI signals
- Update the configuration in main.py

## Usage

Run the app with default settings:

```bash
python main.py
```

### Command-line Options

```
--device N            Camera device index (default: 0)
--midi-port "NAME"    MIDI port name (overrides config in script)
--debug               Enable debug mode with additional face tracking visualization
--list-ports          List available MIDI ports and exit
--threshold N         Threshold for MIDI CC value changes (default: 3)
--theme THEME         UI color theme (choices: dark, light, neon)
```

### Examples

```bash
# Use second camera and neon theme
python main.py --device 1 --theme neon

# List available MIDI ports
python main.py --list-ports

# Use specific MIDI port
python main.py --midi-port "loopMIDI Port 1"

# Reduce MIDI glitches with higher threshold
python main.py --threshold 8
```

## Gesture Controls

### Hand Position Controls

- **Hand Height**: Controls CC 1 (Pitch)
- **Finger Spread**: Controls CC 2 (Formant)
- **Fist Tightness**: Controls CC 4 (Distortion)
- **Palm Angle**: Controls CC 5 (Reverb)

### Facial Controls

- **Mouth Openness**: Controls CC 3 (Vibrato)
- **Eyebrow Raise**: Controls CC 7 (Harmonizer blend)

### Gesture Presets

Make specific hand shapes to activate vocal presets:

- **Index finger only**: Natural voice preset
- **Peace sign (index + middle)**: Robot voice preset
- **Pinky only**: Chipmunk voice preset
- **Closed fist**: Monster voice preset

## Setting Up in DAWs

### Ableton Live

1. Create a new MIDI track
2. Set the MIDI input to your virtual MIDI device
3. Add a vocal effect instrument/plugin (e.g., vocoder, auto-tune)
4. Map the MIDI CC controls to the desired parameters:
   - CC 1 → Pitch
   - CC 2 → Formant
   - CC 3 → Vibrato amount
   - CC 4 → Distortion/Drive
   - CC 5 → Reverb amount

### Cubase

1. Go to Studio > Studio Setup
2. Select "MIDI Port Setup" from the devices list
3. Activate your loopMIDI port
4. Create a MIDI or Instrument track with vocal effects
5. Right-click on effect parameters and use "Learn CC"
6. Make gestures to assign controls

## Troubleshooting

- **No MIDI ports detected**: Ensure loopMIDI is running (Windows) or IAC Driver is enabled (Mac)
- **Wrong MIDI port name**: Check the exact port name with `--list-ports` and update accordingly
- **MIDI glitches**: Increase threshold value with `--threshold` option
- **Poor tracking**: Improve lighting conditions and try `--debug` mode
- **Webcam issues**: Try different camera index with `--device` option
- **CPU performance**: Adjust cooldown value in the code to reduce update frequency

## Customizing

- Edit the `GESTURE_PRESETS` dictionary in the code to create your own vocal presets
- Adjust the color scheme by selecting a different theme (`--theme`)
- Modify the CC mappings at the top of main.py to match your DAW setup
