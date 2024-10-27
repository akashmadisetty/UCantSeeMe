# UCan-tSeeMe

More Like, Can’t See My Personal Info!
 
## Goals

## Approach

## Dependencies
```bash import sys
import os
from datetime import datetime
import logging
from threading import Timer
from PIL import Image, ImageFilter, ImageDraw
import pytesseract
import re
import spacy
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QScreen, QImage 
```

## Demo

## Usage

Follow these steps to set up the project:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/akashmadisetty/UCantSeeMe.git
   ```

2. **Install all the needed modules**
    All the dependencies can be installed using 
    ```bash 
    pip install
    ```
 
3. **Run the Script:**
   Run the `app.py` file, and the PyQt6 interface will start working.