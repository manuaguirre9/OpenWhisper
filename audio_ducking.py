from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class AudioDucker:
    def __init__(self):
        self.original_volume = None
        self.volume_interface = self._get_volume_interface()

    def _get_volume_interface(self):
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return cast(interface, POINTER(IAudioEndpointVolume))
        except Exception as e:
            print(f"[AudioDucker] Failed to get audio interface: {e}")
            return None

    def duck(self, drop_percentage=30):
        """Lowers the system volume by the specified percentage (0 to 100)."""
        if self.volume_interface is None:
            return
            
        try:
            # Get master volume (returns a scalar from 0.0 to 1.0)
            current_vol = self.volume_interface.GetMasterVolumeLevelScalar()
            
            # Save original if not already ducking
            if self.original_volume is None:
                self.original_volume = current_vol
                
            # Calculate new volume
            drop_ratio = drop_percentage / 100.0
            new_vol = max(0.0, current_vol - drop_ratio)
            
            # Apply new volume
            self.volume_interface.SetMasterVolumeLevelScalar(new_vol, None)
            print(f"[AudioDucker] Ducked volume from {current_vol:.2f} to {new_vol:.2f} (Drop: {drop_percentage}%)")
        except Exception as e:
            print(f"[AudioDucker] Error ducking volume: {e}")

    def restore(self):
        """Restores the system volume to its original state before ducking."""
        if self.volume_interface is None or self.original_volume is None:
            return
            
        try:
            self.volume_interface.SetMasterVolumeLevelScalar(self.original_volume, None)
            print(f"[AudioDucker] Restored volume to {self.original_volume:.2f}")
            self.original_volume = None
        except Exception as e:
            print(f"[AudioDucker] Error restoring volume: {e}")
