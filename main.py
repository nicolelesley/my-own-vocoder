import cv2
import mediapipe as mp
import mido
import numpy as np
import time
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='MidiCam Python - Camera-based MIDI controller for vocal effects')
parser.add_argument('--device', type=int, default=0, help='Camera device index (default: 0)')
parser.add_argument('--midi-port', type=str, help='MIDI port name (overrides config in script)')
parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional visual feedback')
parser.add_argument('--list-ports', action='store_true', help='List available MIDI ports and exit')
args = parser.parse_args()

# List MIDI ports if requested
if args.list_ports:
    print("Available MIDI ports:", mido.get_output_names())
    exit(0)

# MIDI Configuration
MIDI_PORT_NAME = 'loopMIDI Port 1'  # Updated by midi_setup.py
# MIDI_PORT_NAME = 'loopMIDI Port'   # For Windows (uncomment if using Windows)

# Override MIDI port if specified in command line
if args.midi_port:
    MIDI_PORT_NAME = args.midi_port

# MIDI CC mappings for vocal effects
CC_PITCH = 1      # Pitch control
CC_FORMANT = 2    # Formant control
CC_VIBRATO = 3    # Vibrato amount
CC_DISTORTION = 4 # Distortion amount
CC_REVERB = 5     # Reverb amount
CC_DELAY = 6      # Delay amount
CC_HARMONIZER = 7 # Harmonizer blend

# Gesture presets - you can customize these
GESTURE_PRESETS = {
    "natural": {"pitch": 64, "formant": 64, "vibrato": 20, "distortion": 0, "reverb": 30},
    "robot": {"pitch": 64, "formant": 100, "vibrato": 0, "distortion": 80, "reverb": 60},
    "chipmunk": {"pitch": 100, "formant": 120, "vibrato": 30, "distortion": 0, "reverb": 40},
    "monster": {"pitch": 30, "formant": 30, "vibrato": 20, "distortion": 100, "reverb": 80}
}

try:
    print("Available MIDI ports:", mido.get_output_names())
    midi_out = mido.open_output(MIDI_PORT_NAME)
    print(f"Connected to MIDI port: {MIDI_PORT_NAME}")
except (IOError, ValueError) as e:
    print(f"Error connecting to MIDI port: {e}")
    print("Available MIDI ports:", mido.get_output_names())
    print("Run with --list-ports to see available options")
    print("Try running midi_setup.py to configure MIDI settings")
    exit(1)

# MediaPipe initialization
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize detectors with appropriate parameters
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(args.device)
hand_detected = False
prev_hand_y = 0
prev_mouth_open = 0
last_gesture_time = time.time()
cooldown = 0.05  # seconds between gesture processing
active_preset = None
note_being_played = None

# For gesture detection history/smoothing
gesture_history = {
    "pitch": [64] * 5,
    "formant": [64] * 5,
    "vibrato": [0] * 5,
    "distortion": [0] * 5,
    "reverb": [30] * 5
}

def apply_smoothing(param_name, new_value, smoothing_factor=0.7):
    """Apply smoothing to parameter values to prevent jitter"""
    # Add the new value to history
    gesture_history[param_name].pop(0)
    gesture_history[param_name].append(new_value)
    
    # Return the weighted average
    return int(sum(gesture_history[param_name]) / len(gesture_history[param_name]))

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def detect_finger_state(hand_landmarks, finger_tip_idx, finger_pip_idx):
    """Detect if a finger is extended (open) or not (closed)"""
    tip = hand_landmarks.landmark[finger_tip_idx]
    pip = hand_landmarks.landmark[finger_pip_idx]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # If the tip is higher than the pip joint (y is smaller), the finger is extended
    # We're checking distance from wrist to make this work in different orientations
    tip_to_wrist = calculate_distance(tip, wrist)
    pip_to_wrist = calculate_distance(pip, wrist)
    
    return tip_to_wrist > pip_to_wrist

def detect_specific_gesture(hand_landmarks):
    """Detect specific hand gestures for preset activation"""
    # Check if each finger is extended
    is_thumb_up = detect_finger_state(
        hand_landmarks, mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP
    )
    is_index_up = detect_finger_state(
        hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP
    )
    is_middle_up = detect_finger_state(
        hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP
    )
    is_ring_up = detect_finger_state(
        hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP
    )
    is_pinky_up = detect_finger_state(
        hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP
    )
    
    # Detect specific gestures
    if is_index_up and not is_middle_up and not is_ring_up and not is_pinky_up:
        # Index finger only - Natural voice preset
        return "natural"
    elif is_index_up and is_middle_up and not is_ring_up and not is_pinky_up:
        # Peace sign - Robot voice preset
        return "robot"
    elif is_pinky_up and not is_index_up and not is_middle_up and not is_ring_up:
        # Pinky only - Chipmunk voice preset
        return "chipmunk"
    elif not is_index_up and not is_middle_up and not is_ring_up and not is_pinky_up:
        # Closed fist - Monster voice preset
        return "monster"
    
    return None

def hand_gesture_processor(hand_landmarks):
    """Process hand landmarks to detect gestures"""
    # Extract key points
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Check for specific gesture presets first
    gesture_preset = detect_specific_gesture(hand_landmarks)
    
    # Vertical position (for pitch)
    y_pos = wrist.y
    pitch_val = int(max(0, min(127, (1 - y_pos) * 127)))
    
    # Calculate finger spread (for formant)
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    spread = (thumb_index_dist * 1000)  # Scale appropriately
    formant_val = int(max(0, min(127, spread)))
    
    # Calculate "fist tightness" (for distortion)
    fist_value = sum([
        calculate_distance(wrist, index_tip),
        calculate_distance(wrist, middle_tip),
        calculate_distance(wrist, ring_tip),
        calculate_distance(wrist, pinky_tip)
    ]) / 4
    distortion_val = 127 - int(max(0, min(127, fist_value * 500)))
    
    # Calculate palm angle (for reverb)
    # Using thumb and pinky as reference points
    dx = thumb_tip.x - pinky_tip.x
    dy = thumb_tip.y - pinky_tip.y
    angle = np.degrees(np.arctan2(dy, dx)) % 180
    reverb_val = int(max(0, min(127, angle / 180 * 127)))
    
    return {
        "pitch": pitch_val,
        "formant": formant_val,
        "distortion": distortion_val,
        "reverb": reverb_val,
        "preset": gesture_preset
    }

def detect_mouth_openness(face_landmarks):
    """Detect how open the mouth is based on face landmarks"""
    if not face_landmarks:
        return 0
    
    # Upper and lower lip landmarks (adjust indices as needed for mediapipe version)
    upper_lip = face_landmarks.landmark[13]  # Upper lip
    lower_lip = face_landmarks.landmark[14]  # Lower lip
    
    # Calculate vertical distance between lips
    mouth_open = abs(upper_lip.y - lower_lip.y)
    
    # Normalize to MIDI range (0-127)
    return int(min(127, mouth_open * 500))

def detect_eyebrow_raise(face_landmarks):
    """Detect eyebrow raise for additional control"""
    if not face_landmarks:
        return 0
    
    # Eyebrow and eye landmarks
    left_eyebrow = face_landmarks.landmark[107]  # Left eyebrow
    left_eye = face_landmarks.landmark[159]      # Left eye
    
    # Vertical distance
    distance = abs(left_eyebrow.y - left_eye.y)
    
    # Normalize to MIDI range (0-127)
    return int(min(127, distance * 1000))

def send_preset_parameters(preset_name):
    """Send MIDI messages for a predefined preset"""
    preset = GESTURE_PRESETS[preset_name]
    
    # Send all CC messages for the preset
    midi_out.send(mido.Message('control_change', control=CC_PITCH, value=preset["pitch"]))
    midi_out.send(mido.Message('control_change', control=CC_FORMANT, value=preset["formant"]))
    midi_out.send(mido.Message('control_change', control=CC_VIBRATO, value=preset["vibrato"]))
    midi_out.send(mido.Message('control_change', control=CC_DISTORTION, value=preset["distortion"]))
    midi_out.send(mido.Message('control_change', control=CC_REVERB, value=preset["reverb"]))
    
    # Return the preset values for display
    return preset

print("Starting MidiCam Python - Press ESC to exit")

# Frame rate calculation
prev_frame_time = 0
new_frame_time = 0
fps = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # FPS calculation
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time
    
    # Mirror image for more intuitive interaction
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hands
    hand_results = hands.process(rgb)
    
    # Process face
    face_results = face_mesh.process(rgb)
    
    # Initialize MIDI values
    pitch_val = 64  # Center/default value
    formant_val = 64
    distortion_val = 0
    vibrato_val = 0
    reverb_val = 30
    eyebrow_raise_val = 0
    
    current_time = time.time()
    
    # Check if enough time has passed since last gesture processing
    if current_time - last_gesture_time > cooldown:
        # Process hand gestures
        if hand_results.multi_hand_landmarks:
            if not hand_detected:
                midi_out.send(mido.Message('note_on', note=60, velocity=100))
                print("Hand detected: MIDI note ON")
                hand_detected = True
                note_being_played = 60
            
            # Process the first detected hand
            hand_data = hand_gesture_processor(hand_results.multi_hand_landmarks[0])
            
            # Check if a preset gesture was detected
            if hand_data["preset"] and hand_data["preset"] != active_preset:
                active_preset = hand_data["preset"]
                preset_values = send_preset_parameters(active_preset)
                pitch_val = preset_values["pitch"]
                formant_val = preset_values["formant"]
                distortion_val = preset_values["distortion"]
                vibrato_val = preset_values["vibrato"]
                reverb_val = preset_values["reverb"]
                print(f"Activated preset: {active_preset}")
            elif not hand_data["preset"]:
                active_preset = None
                # Apply continuous control parameters with smoothing
                pitch_val = apply_smoothing("pitch", hand_data["pitch"])
                formant_val = apply_smoothing("formant", hand_data["formant"])
                distortion_val = apply_smoothing("distortion", hand_data["distortion"])
                reverb_val = apply_smoothing("reverb", hand_data["reverb"])
            
            # Draw hand landmarks
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        else:
            if hand_detected:
                if note_being_played is not None:
                    midi_out.send(mido.Message('note_off', note=note_being_played, velocity=0))
                    note_being_played = None
                print("Hand lost: MIDI note OFF")
                hand_detected = False
                active_preset = None
        
        # Process face gestures
        if face_results.multi_face_landmarks:
            mouth_open = detect_mouth_openness(face_results.multi_face_landmarks[0])
            vibrato_val = apply_smoothing("vibrato", mouth_open)
            
            # Get eyebrow position for additional control (could be mapped to harmonizer)
            eyebrow_raise_val = detect_eyebrow_raise(face_results.multi_face_landmarks[0])
            
            # Draw face landmarks if debug mode is on
            if args.debug:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                    )
        
        # Send MIDI CC messages
        midi_out.send(mido.Message('control_change', control=CC_PITCH, value=pitch_val))
        midi_out.send(mido.Message('control_change', control=CC_FORMANT, value=formant_val))
        midi_out.send(mido.Message('control_change', control=CC_VIBRATO, value=vibrato_val))
        midi_out.send(mido.Message('control_change', control=CC_DISTORTION, value=distortion_val))
        midi_out.send(mido.Message('control_change', control=CC_REVERB, value=reverb_val))
        midi_out.send(mido.Message('control_change', control=CC_HARMONIZER, value=eyebrow_raise_val))
        
        last_gesture_time = current_time
    
    # Display values on screen
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if active_preset:
        cv2.putText(frame, f"PRESET: {active_preset.upper()}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Pitch: {pitch_val}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Formant: {formant_val}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Vibrato: {vibrato_val}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Distortion: {distortion_val}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Reverb: {reverb_val}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display hand gesture guide
    cv2.putText(frame, "Gestures:", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "Index only: Natural", (frame.shape[1] - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, "Peace sign: Robot", (frame.shape[1] - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, "Pinky only: Chipmunk", (frame.shape[1] - 200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, "Fist: Monster", (frame.shape[1] - 200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Show frame
    cv2.imshow('MidiCam Python', frame)
    
    # Check for ESC key press
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Clean up
if note_being_played is not None:
    midi_out.send(mido.Message('note_off', note=note_being_played, velocity=0))

cap.release()
cv2.destroyAllWindows()
midi_out.close()
print("MidiCam Python stopped")
