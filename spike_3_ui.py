import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class FloatingWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        # Make the window frameless, stay on top, and act as a tool (no taskbar icon usually)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        
        # Make the window background transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Make the window transparent for mouse input (click-through)
        # Note: on Windows, this combined with FramelessWindowHint allows clicking right through it!
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Basic setup
        self.setGeometry(100, 100, 200, 100) # x, y, width, height
        
        # Add a label to act as our "icon"
        self.label = QLabel("🎙️ Ready", self)
        self.label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 150); padding: 10px; border-radius: 15px;")
        
        self.label.adjustSize()
        self.resize(self.label.size())

    def update_state(self, state):
        if state == "recording":
            self.label.setText("🔴 Recording...")
            self.label.setStyleSheet("color: white; background-color: rgba(255, 0, 0, 180); padding: 10px; border-radius: 15px;")
        elif state == "processing":
            self.label.setText("⏳ Processing...")
            self.label.setStyleSheet("color: white; background-color: rgba(0, 0, 255, 180); padding: 10px; border-radius: 15px;")
        else:
            self.label.setText("🎙️ Ready")
            self.label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 150); padding: 10px; border-radius: 15px;")
            
        self.label.adjustSize()
        self.resize(self.label.size())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    widget = FloatingWidget()
    widget.show()
    
    print("Floating UI Spike running. You should see a semi-transparent 'Ready' box.")
    print("Press Ctrl+C in the terminal to exit.")
    
    # In a real app, we would have a thread communicating with this UI.
    sys.exit(app.exec())
