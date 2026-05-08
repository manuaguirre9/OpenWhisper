from pynput import keyboard

# Track the state of the modifier keys
ctrl_pressed = False
cmd_pressed = False
is_recording = False

def check_state():
    global is_recording
    if ctrl_pressed and cmd_pressed:
        if not is_recording:
            print("\n[+] Hotkey Ctrl+Windows PRESSED! Recording started...")
            is_recording = True
    else:
        if is_recording:
            print("\n[-] Hotkey Ctrl+Windows RELEASED! Recording stopped.")
            is_recording = False

def on_press(key):
    global ctrl_pressed, cmd_pressed
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        ctrl_pressed = True
    # Key.cmd represents the Windows key on Windows and Command key on Mac
    elif key == keyboard.Key.cmd or key == keyboard.Key.cmd_l or key == keyboard.Key.cmd_r:
        cmd_pressed = True
    
    check_state()

def on_release(key):
    global ctrl_pressed, cmd_pressed
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        ctrl_pressed = False
    elif key == keyboard.Key.cmd or key == keyboard.Key.cmd_l or key == keyboard.Key.cmd_r:
        cmd_pressed = False
    elif key == keyboard.Key.esc:
        print("Exiting...")
        # Stop listener
        return False
        
    check_state()

print("Starting Spike 1: Global Hotkey Detection (using pynput)")
print("Press and hold 'Ctrl + Windows'. Release to stop.")
print("Press 'esc' to exit.")

# Collect events until released
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
