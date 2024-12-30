# Music Stem Splitter Documentation

## Problem Statement
The separation of mixed music into its constituent components (stems) is a fundamental challenge in music production and audio processing. Professional music producers often need isolated tracks for remixing, remastering, or educational purposes. Traditional methods of obtaining stems require access to original multi-track recordings, which are rarely available. This project aims to solve this problem by implementing an automated system that can separate mixed music into individual stems (vocals, drums, bass, and other instruments) using deep learning.

## Methodology

### Machine Learning Approach
We chose the Demucs (Deep Extractor for Music Sources) model developed by Facebook Research for this project. Here's why:

1. **Architecture Justification:**
   - Demucs uses a hybrid approach combining U-Net architecture with bidirectional LSTM
   - Performs in both time and frequency domains for better separation
   - Demonstrates superior performance in multiple source separation benchmarks
   - Handles real-time processing efficiently

2. **Technical Specifications:**
   - Input: Mixed audio signal (stereo/mono)
   - Output: Separated stems (vocals, drums, bass, other)
   - Processing: Combination of convolutional and recurrent layers
   - Sampling Rate: 44.1kHz (standard for music)

### Implementation Steps

1. **Environment Setup:**
```bash
mkdir music_stem_splitter
cd music_stem_splitter
python -m venv venv
source venv/Scripts/activate  # Windows (Git Bash)
```

2. **Dependencies Installation:**
```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

3. **Required Prerequisites:**
   - Python 3.8+ (3.9 recommended)
   - CUDA-compatible GPU (optional, but recommended)
   - Visual Studio Build Tools with C++ support
   - 8GB+ RAM recommended


3. **Quality Characteristics:**
   - Minimal artifacts in separated stems
   - Good preservation of transients
   - Effective handling of overlapping frequencies



## Usage Instructions

1. **Basic Usage:**
```python
from stem_splitter import StemSplitter

splitter = StemSplitter()
stems = splitter.split_tracks("input.mp3", stem_config="4stem")
```

2. **Web Interface:**
```bash
python -m streamlit run app.py
```

3. **Supported Configurations:**
   - 2-stem (vocals/accompaniment)
   - 4-stem (vocals/drums/bass/other)

## Future Improvements

1. **Technical Enhancements:**
   - Add support for more audio formats
   - Implement batch processing for multiple files
   - Add real-time preview functionality

2. **User Experience:**
   - Add progress bars for long processes
   - Implement audio preview before download
   - Add basic audio editing features

## References

1. Défossez, A., et al. (2019). "Music Source Separation in the Waveform Domain"
2. Facebook Research Demucs Repository
3. Streamlit Documentation
4. PyTorch Audio Documentation
