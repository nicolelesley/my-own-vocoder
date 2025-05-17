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
parser.add_argument('--threshold', type=int, default=3, help='Threshold for MIDI CC value changes (default: 3)')
parser.add_argument('--theme', choices=['dark', 'light', 'neon'], default='dark', help='UI color theme')
args = parser.parse_args()

# Choose color theme
if args.theme == 'dark':
    # Dark theme colors
    BG_COLOR = (40, 40, 40)  # Dark gray
    TEXT_COLOR = (240, 240, 240)  # Off-white
    ACCENT_COLOR = (0, 200, 100)  # Green
    HIGHLIGHT_COLOR = (0, 180, 255)  # Orange-ish
    PRESET_COLOR = (180, 180, 0)  # Yellow-ish
    WARNING_COLOR = (0, 0, 220)  # Red
elif args.theme == 'light':
    # Light theme colors
    BG_COLOR = (240, 240, 240)  # Light gray
    TEXT_COLOR = (40, 40, 40)  # Near black
    ACCENT_COLOR = (0, 150, 70)  # Darker green
    HIGHLIGHT_COLOR = (0, 130, 200)  # Blue
    PRESET_COLOR = (130, 100, 0)  # Brown
    WARNING_COLOR = (0, 0, 180)  # Darker red
else:  # neon theme
    # Neon theme colors
    BG_COLOR = (20, 20, 35)  # Very dark blue
    TEXT_COLOR = (200, 255, 200)  # Neon green-white
    ACCENT_COLOR = (0, 255, 170)  # Bright neon green
    HIGHLIGHT_COLOR = (180, 0, 255)  # Neon purple
    PRESET_COLOR = (255, 100, 0)  # Neon orange
    WARNING_COLOR = (0, 60, 255)  # Neon red

# List MIDI ports if requested
if args.list_ports:
    print("Available MIDI ports:", mido.get_output_names())
    exit(0)

# MIDI Configuration
# MIDI_PORT_NAME = 'IAC Driver Bus 1'  # For Mac
MIDI_PORT_NAME = 'loopMIDI Port 1'   # For Windows - using detected port

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

# Create custom drawing specs for better visualization
hand_landmark_style = mp_drawing_styles.get_default_hand_landmarks_style()
hand_connection_style = mp_drawing_styles.get_default_hand_connections_style()

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
cooldown = 0.1  # Increased from 0.05 to 0.1 seconds to reduce MIDI traffic
active_preset = None
note_being_played = None

# For gesture detection history/smoothing
gesture_history = {
    "pitch": [64] * 10,     # Increased history length from 5 to 10
    "formant": [64] * 10,
    "vibrato": [0] * 10,
    "distortion": [0] * 10,
    "reverb": [30] * 10
}

# For tracking previous CC values to filter duplicates
previous_cc_values = {
    CC_PITCH: 64,
    CC_FORMANT: 64,
    CC_VIBRATO: 0,
    CC_DISTORTION: 0,
    CC_REVERB: 30,
    CC_DELAY: 0,
    CC_HARMONIZER: 0
}

def filter_cc_values(control, value, previous_value, threshold=3):
    """Only send CC if value has changed significantly"""
    if abs(value - previous_value) >= threshold:
        midi_out.send(mido.Message('control_change', control=control, value=value))
        return value
    return previous_value

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
    global previous_cc_values  # Use global variable for filtering
    preset = GESTURE_PRESETS[preset_name]
    threshold = args.threshold  # Use command-line threshold
    
    # Send all CC messages for the preset, with filtering to avoid glitches
    previous_cc_values[CC_PITCH] = filter_cc_values(CC_PITCH, preset["pitch"], previous_cc_values[CC_PITCH], threshold)
    previous_cc_values[CC_FORMANT] = filter_cc_values(CC_FORMANT, preset["formant"], previous_cc_values[CC_FORMANT], threshold)
    previous_cc_values[CC_VIBRATO] = filter_cc_values(CC_VIBRATO, preset["vibrato"], previous_cc_values[CC_VIBRATO], threshold)
    previous_cc_values[CC_DISTORTION] = filter_cc_values(CC_DISTORTION, preset["distortion"], previous_cc_values[CC_DISTORTION], threshold)
    previous_cc_values[CC_REVERB] = filter_cc_values(CC_REVERB, preset["reverb"], previous_cc_values[CC_REVERB], threshold)
    
    # Return the preset values for display
    return preset

def draw_progress_bar(frame, x, y, width, value, max_value=127, height=20, color=ACCENT_COLOR, text=""):
    """Draw a modern-looking progress bar"""
    # Calculate fill width based on value
    fill_width = int(width * value / max_value)
    
    # Draw background rectangle
    cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), -1)
    
    # Draw filled part
    cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
    
    # Draw border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)
    
    # Add text label
    if text:
        cv2.putText(frame, f"{text}: {value}", (x + 5, y + height - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

def draw_gesture_guide(frame):
    """Draw the gesture guide panel"""
    height, width = frame.shape[:2]
    panel_width = 200
    panel_height = 150
    x = width - panel_width - 10
    y = 10
    
    # Draw panel background with transparency
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw panel border
    cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), ACCENT_COLOR, 2)
    
    # Draw panel title
    cv2.putText(frame, "Gesture Presets", (x + 10, y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, HIGHLIGHT_COLOR, 2)
    
    # Draw gesture guides
    cv2.putText(frame, "Index only: Natural", (x + 10, y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    cv2.putText(frame, "Peace sign: Robot", (x + 10, y + 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    cv2.putText(frame, "Pinky only: Chipmunk", (x + 10, y + 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    cv2.putText(frame, "Fist: Monster", (x + 10, y + 125), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

def draw_header(frame):
    """Draw a header with app title and status"""
    height, width = frame.shape[:2]
    
    # Draw header background with transparency
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 40), BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw app title
    cv2.putText(frame, "MidiCam Python", (10, 30), 
                cv2.FONT_HERSHEY_TRIPLEX, 1, HIGHLIGHT_COLOR, 2)
    
    # Draw MIDI status
    cv2.putText(frame, f"MIDI: {MIDI_PORT_NAME}", (width - 250, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT_COLOR, 1)

def draw_parameter_panel(frame, pitch_val, formant_val, vibrato_val, distortion_val, reverb_val):
    """Draw a panel with all parameter values and visualizations"""
    height, width = frame.shape[:2]
    panel_x = 10
    panel_y = 50
    panel_width = 250
    panel_height = 200
    
    # Draw panel background with transparency
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw panel border
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), ACCENT_COLOR, 2)
    
    # Draw parameter values as progress bars
    bar_width = panel_width - 20
    bar_x = panel_x + 10
    
    # Draw each parameter bar
    draw_progress_bar(frame, bar_x, panel_y + 30, bar_width, pitch_val, color=ACCENT_COLOR, text="Pitch")
    draw_progress_bar(frame, bar_x, panel_y + 60, bar_width, formant_val, color=HIGHLIGHT_COLOR, text="Formant")
    draw_progress_bar(frame, bar_x, panel_y + 90, bar_width, vibrato_val, color=(ACCENT_COLOR[0], HIGHLIGHT_COLOR[1], ACCENT_COLOR[2]), text="Vibrato")
    draw_progress_bar(frame, bar_x, panel_y + 120, bar_width, distortion_val, color=(HIGHLIGHT_COLOR[0], ACCENT_COLOR[1], HIGHLIGHT_COLOR[2]), text="Distortion")
    draw_progress_bar(frame, bar_x, panel_y + 150, bar_width, reverb_val, color=PRESET_COLOR, text="Reverb")

def draw_preset_indicator(frame, preset_name):
    """Draw a prominent preset indicator"""
    height, width = frame.shape[:2]
    
    # Draw a semi-transparent overlay across the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 40), (width, 100), BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Draw preset name
    cv2.putText(frame, f"PRESET: {preset_name.upper()}", (width//2 - 100, 80), 
                cv2.FONT_HERSHEY_TRIPLEX, 1, PRESET_COLOR, 2)

print("Starting MidiCam Python - Press ESC to exit")
print(f"CC Filtering threshold: {args.threshold}")
print(f"MIDI update rate: {1/cooldown:.1f} messages per second")
print(f"UI Theme: {args.theme}")

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
            
            # Draw hand landmarks with custom styling
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw more prominent hand landmarks
                hand_drawing_spec = mp_drawing.DrawingSpec(
                    color=(HIGHLIGHT_COLOR[2], HIGHLIGHT_COLOR[1], HIGHLIGHT_COLOR[0]),
                    thickness=3,
                    circle_radius=4
                )
                connection_drawing_spec = mp_drawing.DrawingSpec(
                    color=(ACCENT_COLOR[2], ACCENT_COLOR[1], ACCENT_COLOR[0]),
                    thickness=2
                )
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    hand_drawing_spec,
                    connection_drawing_spec
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
                    # Custom face mesh drawing specs
                    contour_spec = mp_drawing.DrawingSpec(
                        color=(HIGHLIGHT_COLOR[2], ACCENT_COLOR[1], HIGHLIGHT_COLOR[0]), 
                        thickness=1, 
                        circle_radius=1
                    )
                    connection_spec = mp_drawing.DrawingSpec(
                        color=(ACCENT_COLOR[2], HIGHLIGHT_COLOR[1], ACCENT_COLOR[0]), 
                        thickness=1, 
                        circle_radius=1
                    )
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        contour_spec,
                        connection_spec
                    )
        
        # Send MIDI CC messages with filtering to avoid glitches
        threshold = args.threshold
        previous_cc_values[CC_PITCH] = filter_cc_values(CC_PITCH, pitch_val, previous_cc_values[CC_PITCH], threshold)
        previous_cc_values[CC_FORMANT] = filter_cc_values(CC_FORMANT, formant_val, previous_cc_values[CC_FORMANT], threshold)
        previous_cc_values[CC_VIBRATO] = filter_cc_values(CC_VIBRATO, vibrato_val, previous_cc_values[CC_VIBRATO], threshold)
        previous_cc_values[CC_DISTORTION] = filter_cc_values(CC_DISTORTION, distortion_val, previous_cc_values[CC_DISTORTION], threshold)
        previous_cc_values[CC_REVERB] = filter_cc_values(CC_REVERB, reverb_val, previous_cc_values[CC_REVERB], threshold)
        previous_cc_values[CC_HARMONIZER] = filter_cc_values(CC_HARMONIZER, eyebrow_raise_val, previous_cc_values[CC_HARMONIZER], threshold)
        
        last_gesture_time = current_time
    
    # Draw header with app title
    draw_header(frame)
    
    # Display FPS in top-right corner
    cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 100, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    # Draw UI elements based on active preset
    if active_preset:
        draw_preset_indicator(frame, active_preset)
    else:
        draw_parameter_panel(frame, pitch_val, formant_val, vibrato_val, distortion_val, reverb_val)
    
    # Draw gesture guide
    draw_gesture_guide(frame)
    
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
