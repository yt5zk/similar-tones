"""
AudioLoader - 音声ファイル読み込みと前処理

WAVおよびOGGファイルを読み込み、CLAP要求仕様に変換する
"""

from pathlib import Path
import numpy as np
from typing import Tuple
import logging

import soundfile as sf
from pydub import AudioSegment
import librosa


logger = logging.getLogger(__name__)


class AudioLoader:
    """
    音声ファイルの読み込みと前処理を担当するクラス
    """
    
    TARGET_SAMPLE_RATE = 48000  # CLAPモデル要求仕様
    TARGET_CHANNELS = 1  # モノラル
    
    def __init__(self):
        """
        AudioLoaderの初期化
        """
        logger.info("AudioLoader initialized")
    
    def load(self, file_path: Path) -> np.ndarray:
        """
        音声ファイルを読み込み、CLAP互換形式に変換
        
        Args:
            file_path: 音声ファイルのパス
            
        Returns:
            48kHz, モノラル, float32のNumPy配列
            
        Raises:
            FileNotFoundError: ファイルが存在しない場合
            ValueError: サポートされていないファイル形式の場合
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        logger.info(f"Loading audio file: {file_path}")
        
        # ファイル拡張子で形式を判定
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.wav':
            # WAVファイルの読み込み
            audio_data, sample_rate = sf.read(str(file_path), dtype='float32')
            logger.debug(f"Loaded WAV: {audio_data.shape}, {sample_rate}Hz")
            
        elif file_extension == '.ogg':
            # OGGファイルの読み込み（pydub経由）
            audio_segment = AudioSegment.from_ogg(str(file_path))
            
            # pydubからNumPy配列に変換
            audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            sample_rate = audio_segment.frame_rate
            
            # ステレオの場合は2列の配列に変換
            if audio_segment.channels == 2:
                audio_data = audio_data.reshape((-1, 2))
            
            # 正規化（16bit整数 -> float32）
            audio_data = audio_data / 32768.0
            
            logger.debug(f"Loaded OGG: {audio_data.shape}, {sample_rate}Hz")
            
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported: .wav, .ogg")
        
        # CLAP要求形式に変換
        processed_audio = self._convert_to_clap_format(audio_data, sample_rate)
        
        logger.info(f"Audio loaded and converted: {processed_audio.shape}, {self.TARGET_SAMPLE_RATE}Hz")
        return processed_audio
    
    def _convert_to_clap_format(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int
    ) -> np.ndarray:
        """
        音声データをCLAP要求形式に変換
        
        Args:
            audio_data: 入力音声データ
            sample_rate: 入力サンプリングレート
            
        Returns:
            変換後の音声データ
        """
        logger.debug(f"Converting audio: {audio_data.shape}, {sample_rate}Hz -> {self.TARGET_SAMPLE_RATE}Hz mono")
        
        # 1. ステレオ -> モノラル変換
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            # ステレオの場合は平均を取ってモノラルに変換
            audio_data = np.mean(audio_data, axis=1)
            logger.debug("Converted stereo to mono")
        
        # audio_dataが1次元でない場合は1次元に変換
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
        
        # 2. サンプリングレート変換
        if sample_rate != self.TARGET_SAMPLE_RATE:
            # librosaでリサンプリング
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sample_rate, 
                target_sr=self.TARGET_SAMPLE_RATE
            )
            logger.debug(f"Resampled from {sample_rate}Hz to {self.TARGET_SAMPLE_RATE}Hz")
        
        # 3. データ型と正規化の確認
        audio_data = audio_data.astype(np.float32)
        
        # 正規化範囲チェック（-1.0 ~ 1.0）
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
            logger.debug(f"Normalized audio data (max was {max_val:.3f})")
        
        logger.debug(f"Final audio shape: {audio_data.shape}, dtype: {audio_data.dtype}")
        return audio_data