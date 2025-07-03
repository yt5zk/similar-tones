"""
AudioLoader - 音声ファイル読み込みと前処理

WAVおよびOGGファイルを読み込み、CLAP要求仕様に変換する
"""

from pathlib import Path
import numpy as np
from typing import Tuple
import logging

# TODO: 以下のインポートは後のステップで実装
# import soundfile as sf
# from pydub import AudioSegment
# import librosa


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
        
        # TODO: 実装
        # 1. ファイル形式を判定（WAV/OGG）
        # 2. 適切なライブラリで読み込み
        # 3. サンプリングレート変換（44.1kHz -> 48kHz）
        # 4. モノラル変換（ステレオ -> モノラル）
        # 5. データ型変換（int16 -> float32, 正規化 -1.0~1.0）
        
        raise NotImplementedError("load method not implemented yet")
    
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
        # TODO: 実装
        # リサンプリング、モノラル変換、正規化処理
        raise NotImplementedError("_convert_to_clap_format method not implemented yet")