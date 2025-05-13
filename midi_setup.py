import mido
import sys
import time

def list_midi_ports():
    """List all available MIDI input and output ports"""
    print("\n=== Available MIDI Output Ports ===")
    for port in mido.get_output_names():
        print(f"- {port}")
    
    print("\n=== Available MIDI Input Ports ===")
    for port in mido.get_input_names():
        print(f"- {port}")

def test_midi_connection(port_name):
    """Test MIDI connection by sending a series of notes"""
    try:
        port = mido.open_output(port_name)
        print(f"\nSuccessfully connected to: {port_name}")
        
        # Send a series of test notes
        print("Sending test MIDI signals (5 notes)...")
        for note in [60, 64, 67, 72, 60]:
            port.send(mido.Message('note_on', note=note, velocity=100))
            time.sleep(0.2)
            port.send(mido.Message('note_off', note=note, velocity=0))
            time.sleep(0.1)
        
        # Send CC messages
        print("Sending test MIDI CC messages...")
        for cc in range(1, 6):
            for val in [0, 64, 127, 64, 0]:
                port.send(mido.Message('control_change', control=cc, value=val))
                time.sleep(0.05)
        
        port.close()
        print("Test completed successfully!")
        return True
    
    except (IOError, ValueError) as e:
        print(f"Error connecting to {port_name}: {e}")
        return False

def update_main_script(port_name):
    """Update the MIDI_PORT_NAME in main.py"""
    try:
        with open('main.py', 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            if "MIDI_PORT_NAME" in line and "=" in line:
                lines[i] = f"MIDI_PORT_NAME = '{port_name}'  # Updated by midi_setup.py\n"
                break
        
        with open('main.py', 'w') as f:
            f.writelines(lines)
        
        print(f"\nUpdated main.py with MIDI port: {port_name}")
        return True
    
    except Exception as e:
        print(f"Error updating main.py: {e}")
        return False

def main():
    print("=== MIDI Setup Utility for MidiCam Python ===")
    
    # List available ports
    list_midi_ports()
    
    # Ask user to select a port
    print("\nSelect a MIDI output port to use with MidiCam Python")
    port_name = input("Enter the exact name of the MIDI port: ")
    
    # Test the connection
    if test_midi_connection(port_name):
        # Update the main script
        if update_main_script(port_name):
            print("\nSetup complete! You can now run main.py to start the application.")
    else:
        print("\nFailed to connect to the specified MIDI port.")
        print("Please check your MIDI configuration and try again.")

if __name__ == "__main__":
    main() 