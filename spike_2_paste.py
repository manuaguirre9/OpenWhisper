import pyperclip
import pyautogui
import time
import keyboard

def inject_text(text_to_inject):
    print(f"\n[+] Injecting text: '{text_to_inject}'")
    
    # 1. Backup current clipboard
    original_clipboard = pyperclip.paste()
    print(f"    - Backed up clipboard: '{original_clipboard}'")
    
    # 2. Put new text in clipboard
    pyperclip.copy(text_to_inject)
    print("    - Copied new text to clipboard.")
    
    # Wait a tiny bit for OS clipboard to register
    time.sleep(0.1)
    
    # 3. Simulate Ctrl+V
    print("    - Simulating Ctrl+V...")
    pyautogui.hotkey('ctrl', 'v')
    
    # Wait for the paste to actually happen before restoring clipboard
    time.sleep(0.2)
    
    # 4. Restore original clipboard
    pyperclip.copy(original_clipboard)
    print("    - Restored original clipboard.")

print("Starting Spike 2: Clipboard Paste Injection")
print("Open a text editor (like Notepad), click inside it, and press F8 to trigger a paste.")
print("Press 'esc' to exit.")

try:
    while True:
        if keyboard.is_pressed('f8'):
            # Wait for the key to be released so it doesn't trigger multiple times
            while keyboard.is_pressed('f8'):
                time.sleep(0.05)
            
            # Inject a test string
            inject_text("Hello from the WhisprFlow MVP! This was pasted instantly.")
            
        if keyboard.is_pressed('esc'):
            print("Exiting...")
            break
            
        time.sleep(0.05)
except KeyboardInterrupt:
    print("Exiting...")
