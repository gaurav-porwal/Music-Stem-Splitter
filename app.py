import streamlit as st
import torch
import torchaudio
import os
import numpy as np
from demucs.pretrained import get_model
from demucs.apply import apply_model
import soundfile as sf
import tempfile
from pathlib import Path
import shutil

class StemSplitter:
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self, model_name="htdemucs"):
        """Load the Demucs model."""
        self.model = get_model(model_name)
        self.model.to(self.device)
        
    def split_tracks(self, audio_path, stem_config="4stem"):
        """
        Split the audio file into stems.
        Args:
            audio_path: Path to input audio file
            stem_config: "2stem" (vocals/accompaniment) or "4stem" (vocals/drums/bass/other)
        Returns:
            Dictionary of separated stems
        """
        # Load audio
        wav, sr = torchaudio.load(audio_path)
        wav = wav.to(self.device)
        
        # Apply model
        with torch.no_grad():
            stems = apply_model(self.model, wav, device=self.device)
        
        # Convert to numpy and organize stems
        stems_dict = {}
        source_names = self.model.sources
        
        if stem_config == "2stem":
            # Combine all non-vocal stems for accompaniment
            vocals_idx = source_names.index("vocals")
            stems_dict["vocals"] = stems[vocals_idx].cpu().numpy()
            stems_dict["accompaniment"] = np.sum([stems[i].cpu().numpy() 
                                                for i in range(len(source_names)) 
                                                if i != vocals_idx], axis=0)
        else:
            # 4-stem separation
            for i, name in enumerate(source_names):
                stems_dict[name] = stems[i].cpu().numpy()
                
        return stems_dict, sr

def save_stem(stem_data, sr, output_path):
    """Save a stem to disk."""
    sf.write(output_path, stem_data.T, sr)

# Streamlit UI
st.title("Music Stem Splitter")
st.write("""
Upload an audio file to separate it into individual stems using the Demucs model.
Choose between 2-stem (vocals/accompaniment) or 4-stem (vocals/drums/bass/other) separation.
""")

# File upload
uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav'])

# Stem configuration selection
stem_config = st.radio(
    "Select stem configuration:",
    ("2stem", "4stem"),
    help="2-stem splits into vocals and accompaniment. 4-stem splits into vocals, drums, bass, and other."
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Initialize and load model
    splitter = StemSplitter()
    with st.spinner("Loading model..."):
        splitter.load_model()
    
    # Process audio
    if st.button("Split Audio"):
        with st.spinner("Processing audio..."):
            stems, sr = splitter.split_tracks(tmp_path, stem_config)
            
            # Create temporary directory for stems
            temp_dir = tempfile.mkdtemp()
            
            # Save stems and create download buttons
            st.write("### Download Stems")
            for stem_name, stem_data in stems.items():
                output_path = os.path.join(temp_dir, f"{stem_name}.wav")
                save_stem(stem_data, sr, output_path)
                
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label=f"Download {stem_name.title()} Stem",
                        data=f,
                        file_name=f"{stem_name}.wav",
                        mime="audio/wav"
                    )
            
            # Cleanup
            os.unlink(tmp_path)
            shutil.rmtree(temp_dir)