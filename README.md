Here's the README formatted as markdown:

# Audio Command Recognition and Action Performer

This project implements a system for recognizing audio commands and performing corresponding actions in a graphical user interface. It uses a combination of deep learning and dynamic time warping (DTW) for audio recognition.

## Features

- Audio recording and playback
- Voice command recognition
- GUI for visualizing actions
- Ability to train new words
- Real-time shape drawing and animation based on voice commands

## Components

1. **SoundModel**: A PyTorch-based convolutional neural network for audio classification.

2. **DTW (Dynamic Time Warping)**: An algorithm for measuring similarity between temporal sequences.

3. **SR_System (Speech Recognition System)**: Combines the SoundModel and DTW for robust command recognition.

4. **AudioRecorder**: Handles audio recording and playback using `sounddevice`.

5. **ActionPerformer**: Executes graphical actions based on recognized commands.

6. **GUI**: Tkinter-based user interface for interaction with the system.

## Setup

1. Install required dependencies:
   ```
   pip install torch numpy scipy librosa sounddevice matplotlib tkinter
   ```

2. Ensure you have the following files in your project directory:
   - `model.py` (contains SoundModel)
   - `dtw_utils.py` (contains DTW implementation)
   - `sr_system.py` (contains SR_System)
   - `main.py` (contains GUI, AudioRecorder, and ActionPerformer)

3. Prepare a dataset of audio commands and train the SoundModel (training code not provided in the snippets).

4. Generate a `commands.json` file with the mapping of commands to indices.

## Usage

1. Run the main script:
   ```
   python main.py
   ```

2. Use the GUI to:
   - Record audio commands
   - Play back recorded audio
   - Execute recognized commands
   - Train new words

3. Supported commands (default):
   - "kvadrat" (square)
   - "krug" (circle)
   - "trougao" (triangle)
   - "oboji" (color)
   - "izbrisi" (delete)
   - "animiraj" (animate) - can be added as a new command
