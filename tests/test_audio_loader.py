"""
AudioLoaderのユニットテスト

WAV/OGG読み込み、リサンプリング、モノラル変換のテスト
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from pydub import AudioSegment
import soundfile as sf

from src.audio_loader import AudioLoader


class TestAudioLoader:
    """AudioLoaderのテストクラス"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.audio_loader = AudioLoader()
        self.test_files = []  # 後でクリーンアップ用
    
    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ"""
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def _create_test_wav(
        self, 
        sample_rate: int = 44100, 
        channels: int = 1, 
        duration: float = 0.1,
        frequency: float = 440.0
    ) -> Path:
        """テスト用WAVファイルを作成"""
        # サイン波を生成
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # ステレオの場合
        if channels == 2:
            wave = np.column_stack([wave, wave * 0.8])  # 右チャンネルを少し小さく
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            file_path = Path(temp_file.name)
            sf.write(str(file_path), wave, sample_rate)
        
        self.test_files.append(str(file_path))
        return file_path
    
    def _create_test_ogg(
        self, 
        sample_rate: int = 44100, 
        channels: int = 1, 
        duration: float = 0.1,
        frequency: float = 440.0
    ) -> Path:
        """テスト用OGGファイルを作成"""
        # まずWAVを作成
        wav_path = self._create_test_wav(sample_rate, channels, duration, frequency)
        
        # OGGに変換
        audio = AudioSegment.from_wav(str(wav_path))
        
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
            ogg_path = Path(temp_file.name)
            audio.export(str(ogg_path), format="ogg")
        
        self.test_files.append(str(ogg_path))
        return ogg_path
    
    def test_load_wav_mono_44khz(self):
        """44.1kHz モノラル WAVファイルの読み込みテスト"""
        # テストファイル作成
        wav_path = self._create_test_wav(sample_rate=44100, channels=1, duration=0.1)
        
        # 読み込み
        result = self.audio_loader.load(wav_path)
        
        # 検証
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result.shape) == 1  # 1次元配列（モノラル）
        assert len(result) == int(0.1 * 48000)  # 48kHzに変換されている
        assert -1.0 <= result.min() <= result.max() <= 1.0  # 正規化範囲
    
    def test_load_wav_stereo_48khz(self):
        """48kHz ステレオ WAVファイルの読み込みテスト"""
        # テストファイル作成
        wav_path = self._create_test_wav(sample_rate=48000, channels=2, duration=0.1)
        
        # 読み込み
        result = self.audio_loader.load(wav_path)
        
        # 検証
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result.shape) == 1  # モノラルに変換
        assert len(result) == int(0.1 * 48000)  # 48kHz維持
        assert -1.0 <= result.min() <= result.max() <= 1.0
    
    def test_load_ogg_mono(self):
        """OGGファイルの読み込みテスト"""
        # テストファイル作成
        ogg_path = self._create_test_ogg(sample_rate=44100, channels=1, duration=0.1)
        
        # 読み込み
        result = self.audio_loader.load(ogg_path)
        
        # 検証
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result.shape) == 1
        # OGGの場合、圧縮による誤差があるため範囲を緩く設定
        expected_length = int(0.1 * 48000)
        assert abs(len(result) - expected_length) < 200  # 200サンプル以内の誤差
        assert -1.0 <= result.min() <= result.max() <= 1.0
    
    def test_load_ogg_stereo(self):
        """ステレオOGGファイルの読み込みテスト"""
        # テストファイル作成
        ogg_path = self._create_test_ogg(sample_rate=44100, channels=2, duration=0.1)
        
        # 読み込み
        result = self.audio_loader.load(ogg_path)
        
        # 検証
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result.shape) == 1  # モノラルに変換
        expected_length = int(0.1 * 48000)
        assert abs(len(result) - expected_length) < 200
        assert -1.0 <= result.min() <= result.max() <= 1.0
    
    def test_file_not_found(self):
        """存在しないファイルのエラーテスト"""
        non_existent_path = Path("/non/existent/file.wav")
        
        with pytest.raises(FileNotFoundError):
            self.audio_loader.load(non_existent_path)
    
    def test_unsupported_format(self):
        """サポートされていないファイル形式のエラーテスト"""
        # 空のmp3ファイルを作成
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            mp3_path = Path(temp_file.name)
        
        self.test_files.append(str(mp3_path))
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            self.audio_loader.load(mp3_path)
    
    def test_convert_to_clap_format_resampling(self):
        """_convert_to_clap_formatメソッドのリサンプリングテスト"""
        # 22kHzのテストデータ
        sample_rate = 22050
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # 変換実行
        result = self.audio_loader._convert_to_clap_format(audio_data, sample_rate)
        
        # 検証
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result.shape) == 1
        # 48kHzに変換されているか
        expected_length = int(0.1 * 48000)
        assert abs(len(result) - expected_length) < 10  # 10サンプル以内の誤差
    
    def test_convert_to_clap_format_stereo_to_mono(self):
        """_convert_to_clap_formatメソッドのステレオ→モノラル変換テスト"""
        # ステレオテストデータ
        sample_rate = 48000
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = np.sin(2 * np.pi * 880 * t).astype(np.float32)  # 異なる周波数
        stereo_data = np.column_stack([left, right])
        
        # 変換実行
        result = self.audio_loader._convert_to_clap_format(stereo_data, sample_rate)
        
        # 検証
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result.shape) == 1  # モノラルに変換
        assert len(result) == len(left)  # 同じ長さ
        
        # モノラル変換が正しく実行されているか（左右の平均）
        expected_mono = (left + right) / 2
        np.testing.assert_array_almost_equal(result, expected_mono, decimal=5)
    
    def test_convert_to_clap_format_normalization(self):
        """_convert_to_clap_formatメソッドの正規化テスト"""
        # 範囲外のテストデータ（-2.0 ~ 2.0）
        sample_rate = 48000
        audio_data = np.array([2.0, -2.0, 1.5, -1.5, 0.0], dtype=np.float32)
        
        # 変換実行
        result = self.audio_loader._convert_to_clap_format(audio_data, sample_rate)
        
        # 検証
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert -1.0 <= result.min() <= result.max() <= 1.0  # 正規化されている
        
        # 正規化が正しく実行されているか
        expected = audio_data / 2.0  # 最大値2.0で正規化
        np.testing.assert_array_almost_equal(result, expected, decimal=5)