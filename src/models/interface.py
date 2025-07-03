"""
EmbeddingModelInterface - 特徴抽出モデルのインターフェース

将来的なモデル差し替えに対応するためのプラグイン構造
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class EmbeddingModelInterface(ABC):
    """
    音響特徴量抽出モデルの抽象基底クラス
    """
    
    @abstractmethod
    def get_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """
        音声データから埋め込みベクトルを生成
        
        Args:
            audio_data: 48kHz, モノラル, float32の音声データ
            
        Returns:
            埋め込みベクトル（NumPy配列）
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        埋め込みベクトルの次元数を取得
        
        Returns:
            ベクトルの次元数
        """
        pass