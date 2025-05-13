# Setting Up MidiCam Python with Ableton Live

This guide will help you configure Ableton Live to work with the MidiCam Python application for vocal effects control.

## Prerequisites

- Ableton Live (10 or 11)
- Virtual MIDI port configured (IAC Driver on Mac or loopMIDI on Windows)
- MidiCam Python application running and connected to your virtual MIDI port

## Basic Setup

1. **Launch Ableton Live** and create a new project.

2. **Configure MIDI Settings**:
   - Go to Preferences > Link/MIDI
   - In the MIDI Ports section, enable "Track" and "Remote" for your virtual MIDI port
   - Click "Save"

3. **Create a New MIDI Track**:
   - Click the "+" icon in the session view to add a new MIDI track
   - Set the MIDI From dropdown to your virtual MIDI port (IAC Driver or loopMIDI)
   - Set Monitor to "In"

## Vocal Effects Setup

### Option 1: Using Ableton's Built-in Vocal Effects

1. **Add an Audio Track** for your microphone input.
2. **Add the following effects** to your vocal chain:
   - Auto Filter (for formant shifting)
   - Pitch Shifter
   - Vocoder
   - Audio Effect Rack (to organize multiple effects)

3. **Map MIDI Controllers**:
   - Enter MIDI Map Mode (Ctrl+M/Cmd+M)
   - Click on effect parameters you want to control
   - Move any slider in the MidiCam Python interface to assign controls
   - Exit MIDI Map Mode when done

### Option 2: Using Third-party Vocal Plugins

1. **Recommended Plugins**:
   - iZotope VocalSynth 2 (excellent for comprehensive vocal processing)
   - Antares Auto-Tune
   - Waves OVox
   - Native Instruments The Mouth

2. **Example Setup with iZotope VocalSynth 2**:
   - Insert VocalSynth 2 on an audio track with your mic input
   - Open the plugin interface
   - Right-click on parameters to assign MIDI CC numbers:
     - Pitch: CC 1
     - Formant: CC 2 
     - Distortion: CC 4
     - Reverb Mix: CC 5
     - Vocal Wideness: CC 6
     - Harmonizer: CC 7

## Advanced MIDI Mapping

For more precise control, you can create an Audio Effect Rack with macro controls:

1. **Create an Audio Effect Rack** on your vocal track.
2. **Add various effects** inside the rack.
3. **Map effect parameters** to the 8 macro knobs.
4. **In MIDI Map Mode**, assign the macro knobs to CCs 1-8 from your MidiCam Python app.

This approach gives you more flexibility, as you can assign multiple parameters to a single gesture.

## Preset System

The MidiCam Python app includes gesture presets (Natural, Robot, Chipmunk, Monster). For the best experience:

1. **Create corresponding presets** in Ableton Live or your vocal plugin.
2. **Use Ableton's Clip Envelopes** to automate plugin parameters.
3. **Trigger clips** using Note On/Off messages from the app (note 60).

## Troubleshooting

- **No MIDI Input**: Make sure your virtual MIDI port is properly configured and selected in Ableton
- **Delayed Response**: Lower the buffer size in Ableton's audio preferences
- **CPU Issues**: Increase the cooldown value in the MidiCam Python app
- **Poor Tracking**: Improve lighting conditions and camera positioning

## Tips for Performance

- **Calibrate your setup** by adjusting the scaling factors in the Python code
- **Use a pop filter** on your microphone to avoid unwanted triggers
- **Find a consistent position** for your camera to ensure reliable tracking
- **Combine with Ableton Push or other controllers** for additional performance options 