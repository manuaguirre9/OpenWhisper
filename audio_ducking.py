from comtypes import CoInitialize
from pycaw.pycaw import AudioUtilities

class AudioDucker:
    def __init__(self):
        self.original_volume = None

    def _get_volume(self):
        """Fetches the master volume interface for the current thread."""
        try:
            CoInitialize()
            devices = AudioUtilities.GetSpeakers()
            if devices is None:
                return None
            return devices.EndpointVolume
        except Exception as e:
            print(f"[AudioDucker] Failed to get audio interface: {e}")
            return None

    def duck(self, drop_percentage=30):
        """Lowers the system volume by the specified percentage (0 to 100)."""
        volume = self._get_volume()
        if volume is None:
            return
            
        try:
            current_vol = volume.GetMasterVolumeLevelScalar()
            
            # Save original if not already ducking
            if self.original_volume is None:
                self.original_volume = current_vol
                
            drop_ratio = drop_percentage / 100.0
            new_vol = max(0.0, current_vol - drop_ratio)
            
            volume.SetMasterVolumeLevelScalar(new_vol, None)
            print(f"[AudioDucker] Ducked volume from {current_vol:.2f} to {new_vol:.2f} (Drop: {drop_percentage}%)")
        except Exception as e:
            print(f"[AudioDucker] Error ducking volume: {e}")

    def restore(self):
        """Restores the system volume to its original state before ducking."""
        volume = self._get_volume()
        if volume is None or self.original_volume is None:
            return
            
        try:
            volume.SetMasterVolumeLevelScalar(self.original_volume, None)
            print(f"[AudioDucker] Restored volume to {self.original_volume:.2f}")
            self.original_volume = None
        except Exception as e:
            print(f"[AudioDucker] Error restoring volume: {e}")
