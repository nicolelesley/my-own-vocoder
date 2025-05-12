import cv2
import mediapipe as mp
import mido

MIDI_PORT_NAME = 'IAC Driver Bus 1'

midi_out = midi.open_output(MIDI_PORT_NAME)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)
hand_detected = False

while cap.isOpened();
  ret, frame = cap.read()
  if not ret:
    break

  frame = cv2.flip(frame, 1)
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  hand_Results = hands.process(rgb)

  if hand_results.multi_hand_landmarks:
    if not hand_detected:
      midi_out.send(mido.Message('note_on', note=60, velocity=100))
      print("Hand detected: MIDI note ON")
      hand_detected = True
  else:
    if hand_detected:
      midi_out.send(midi.Message('not_off', note=60, velocity=0))
      print("Hand lost: MIDI note OFF")
      hand_detected = False

  cv2.imshow('MidiCam Python', frame)
  if cv2.waitKey(5) & 0xFF == 27:
    break

cap.release()
cv2.destroyAllWindows()
