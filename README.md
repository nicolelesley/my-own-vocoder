# MidiCam Python

A Python-based alternative to MidiCam that captures hand and facial gestures via webcam and converts them to MIDI signals for vocal production in Ableton Live or other DAWs.

## Features

- Hand gesture detection for controlling various vocal parameters
- Facial gesture recognition for additional control
- Real-time MIDI control change messages
- Visual feedback of detected gestures and parameter values
- Configurable MIDI port and controller mappings

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
pip install opencv-python mediapipe mido numpy python-rtmidi
```

2. Set up a virtual MIDI device:
   - **Mac**: Enable the IAC Driver in Audio MIDI Setup
   - **Windows**: Install and configure [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html)

3. Configure your DAW (Ableton Live) to receive MIDI from the virtual MIDI port

## Usage

1. Edit the `MIDI_PORT_NAME` in the script to match your virtual MIDI device
2. Run the script:

```bash
python main.py
```

3. The application will display your webcam feed with hand and face tracking overlays
4. Position yourself in front of the camera and observe the MIDI controls on screen
5. Press ESC to exit

## MIDI Control Mappings

- **Hand Detection**: Note On/Off (note 60)
- **Hand Height**: CC 1 (Pitch)
- **Finger Spread**: CC 2 (Formant)
- **Mouth Openness**: CC 3 (Vibrato)
- **Fist Tightness**: CC 4 (Distortion)

## Setting Up in Ableton Live

1. Create a new MIDI track
2. Set the MIDI input to your virtual MIDI device
3. Add a vocal effect instrument/plugin (e.g., vocoder, auto-tune)
4. Map the MIDI CC controls to the desired parameters in your vocal plugin:
   - CC 1 → Pitch
   - CC 2 → Formant
   - CC 3 → Vibrato amount
   - CC 4 → Distortion/Drive

5. Arm the track for recording to monitor the MIDI input

## Troubleshooting

- If no MIDI ports are detected, check that your virtual MIDI device is properly configured
- If the webcam doesn't open, ensure no other applications are using it
- For optimal hand tracking, ensure good lighting conditions
- Adjust the `min_detection_confidence` values if tracking is inconsistent
