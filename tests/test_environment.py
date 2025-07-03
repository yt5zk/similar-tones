"""
Environment verification tests for Similar Tones Application.

This module tests that all required dependencies and hardware are properly configured:
- PyTorch and GPU availability
- CLAP model loading from Hugging Face
- Audio file processing (OGG decoding with ffmpeg)
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from transformers import ClapModel, ClapProcessor
import tempfile
import os


class TestEnvironment:
    """Test suite for environment verification."""

    def test_pytorch_gpu_availability(self):
        """Test that PyTorch can detect GPU if available."""
        # This test passes whether GPU is available or not
        # It's mainly to verify PyTorch installation
        gpu_available = torch.cuda.is_available()
        print(f"GPU available: {gpu_available}")
        
        if gpu_available:
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
        
        # Test passes regardless of GPU availability
        assert torch.tensor([1.0]).numpy()[0] == 1.0

    def test_clap_model_loading(self):
        """Test that CLAP model can be loaded from Hugging Face."""
        try:
            # Load CLAP model and processor
            model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
            processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            
            # Verify model components exist
            assert model is not None
            assert processor is not None
            assert hasattr(model, 'get_audio_features')
            
            print("CLAP model loaded successfully")
            
        except Exception as e:
            pytest.fail(f"Failed to load CLAP model: {e}")

    def test_audio_processing_with_dummy_ogg(self):
        """Test OGG file processing using pydub (ffmpeg backend)."""
        # Create a temporary OGG file for testing
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
            temp_ogg_path = temp_file.name
        
        try:
            # Create a simple sine wave audio (1 second, 44.1kHz)
            sample_rate = 44100
            duration = 1.0  # seconds
            frequency = 440.0  # A note
            
            # Generate sine wave
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = np.sin(2 * np.pi * frequency * t)
            
            # Convert to 16-bit integer
            wave = (wave * 32767).astype(np.int16)
            
            # Create AudioSegment and export as OGG
            audio = AudioSegment(
                wave.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit = 2 bytes
                channels=1
            )
            
            # Export as OGG (this tests ffmpeg integration)
            audio.export(temp_ogg_path, format="ogg")
            
            # Verify file was created
            assert os.path.exists(temp_ogg_path)
            assert os.path.getsize(temp_ogg_path) > 0
            
            # Test loading the OGG file back
            loaded_audio = AudioSegment.from_ogg(temp_ogg_path)
            
            # Verify basic properties
            assert loaded_audio.frame_rate == sample_rate
            assert loaded_audio.channels == 1
            assert len(loaded_audio) > 900  # Should be around 1000ms
            
            print(f"OGG processing test successful: {len(loaded_audio)}ms audio")
            
        except Exception as e:
            pytest.fail(f"Failed to process OGG file: {e}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_ogg_path):
                os.unlink(temp_ogg_path)

    def test_numpy_audio_conversion(self):
        """Test conversion of audio data to numpy arrays as required by CLAP."""
        # Create sample audio data
        sample_rate = 48000
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Test array properties required for CLAP
        assert audio_data.dtype == np.float32
        assert len(audio_data.shape) == 1  # Should be 1D array
        assert -1.0 <= audio_data.min() <= audio_data.max() <= 1.0
        
        print(f"Audio array shape: {audio_data.shape}, dtype: {audio_data.dtype}")
        print(f"Audio range: [{audio_data.min():.3f}, {audio_data.max():.3f}]")

    def test_dependency_imports(self):
        """Test that all required dependencies can be imported."""
        import torch
        import torchaudio
        import transformers
        import librosa
        import soundfile
        import numpy
        import pydub
        import typer
        
        # Test version requirements (basic check)
        assert hasattr(torch, 'cuda')
        assert hasattr(transformers, 'ClapModel')
        
        print("All required dependencies imported successfully")