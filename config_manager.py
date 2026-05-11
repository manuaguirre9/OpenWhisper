import json
import os

app_data = os.environ.get('APPDATA')
if app_data:
    CONFIG_DIR = os.path.join(app_data, "OpenWhisper")
else:
    CONFIG_DIR = os.path.join(os.path.expanduser('~'), ".openwhisper")

os.makedirs(CONFIG_DIR, exist_ok=True)
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

DEFAULT_CONFIG = {
    "language": "es",
    "model_size": "base",
    "microphone": "default",
    "hotkey": "ctrl+windows",
    "ducking_percentage": 30
}

def load_config():
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
            # Merge with defaults in case of missing keys
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
    except Exception as e:
        print(f"Error loading config: {e}. Using defaults.")
        return DEFAULT_CONFIG.copy()

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")
