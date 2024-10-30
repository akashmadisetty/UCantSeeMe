# UCan'tSeeMe 🕶️

Privacy when you need it, visibility when you want it. UCan'tSeeMe is your intelligent privacy shield that automatically detects and blurs sensitive information during screen sharing sessions.


## What's This All About? 

Ever been in a Teams meeting and suddenly realized your password manager is visible? Or about to share your screen with sensitive emails in view? UCan'tSeeMe has got your back! It's like having a privacy bodyguard that works in real-time to protect your sensitive information during screen sharing.

## Features 

- **Real-time Protection**: Instant blurring of sensitive information
- **Smart Detection**: AI-powered recognition of sensitive content
- **Conference Ready**: Works with:
  - Microsoft Teams
  - And any other screen sharing platform!
- **User-Friendly**: Simple Qt-based interface that stays out of your way
- **Local Processing**: All detection happens on your machine - what happens on your screen, stays on your screen

## Quick Start 

### Prerequisites
```bash
Python 3.8+
Tesseract OCR
```

### Installation 

1. Clone your new privacy buddy:
```bash
git clone https://github.com/akashmadisetty/UCantSeeMe.git
cd UCantSeeMe
```

2. Set up your privacy fortress:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install the privacy tools:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Usage 
1. Launch your privacy shield:
```bash
python src/main.py
```
3. Share your screen with confidence! 

## Under the Hood 

### Dependencies
```plaintext
🖼️ Pillow>=10.0.0          # Image processing wizardry
📝 pytesseract>=0.3.10     # Text detection magic
🧠 spacy>=3.7.2           # Natural language understanding
🔢 numpy>=1.24.0          # Number crunching power
👁️ opencv-python>=4.8.0    # Computer vision capabilities
🤖 huggingface-hub>=0.19.0 # AI model management
🎨 PyQt6>=6.6.0           # Pretty UI framework
📊 doclayout-yolo>=1.0.0   # Document layout analysis
```
