"""
ClapModel - CLAP特徴抽出モデルの実装

EmbeddingModelInterfaceの具体実装
"""

import numpy as np
import logging
from typing import Optional
from .interface import EmbeddingModelInterface

# TODO: 以下のインポートは後のステップで実装
# import torch
# from transformers import ClapModel as HFClapModel, ClapProcessor


logger = logging.getLogger(__name__)


class ClapModel(EmbeddingModelInterface):
    """
    CLAP（Contrastive Language-Audio Pre-training）モデルの実装
    """
    
    MODEL_NAME = "laion/clap-htsat-unfused"
    EMBEDDING_DIMENSION = 512  # CLAPの埋め込み次元
    
    def __init__(self, device: Optional[str] = None):
        """
        CLAPモデルの初期化
        
        Args:
            device: 使用するデバイス（"cuda", "cpu", None=auto）
        """
        self.device = device
        self.model = None
        self.processor = None
        
        logger.info(f"Initializing CLAP model: {self.MODEL_NAME}")
        self._load_model()
    
    def _load_model(self):
        """
        Hugging FaceからCLAPモデルをロード
        """
        # TODO: 実装
        # 1. transformers.ClapModel.from_pretrained() でモデルロード
        # 2. transformers.ClapProcessor.from_pretrained() でプロセッサロード
        # 3. デバイス設定（GPU/CPU）
        # 4. モデルを評価モードに設定
        
        logger.info("CLAP model loading is not implemented yet")
        # self.model = HFClapModel.from_pretrained(self.MODEL_NAME)
        # self.processor = ClapProcessor.from_pretrained(self.MODEL_NAME)
    
    def get_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """
        音声データからCLAP埋め込みベクトルを生成
        
        Args:
            audio_data: 48kHz, モノラル, float32の音声データ
            
        Returns:
            512次元の埋め込みベクトル
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("CLAP model not loaded")
        
        logger.debug(f"Generating embedding for audio data: {audio_data.shape}")
        
        # TODO: 実装
        # 1. プロセッサで音声データを前処理
        # 2. モデルで推論実行
        # 3. 音声特徴量を抽出
        # 4. NumPy配列として返却
        
        raise NotImplementedError("get_embedding method not implemented yet")
    
    def get_embedding_dimension(self) -> int:
        """
        CLAP埋め込みベクトルの次元数を取得
        
        Returns:
            512（CLAPの埋め込み次元）
        """
        return self.EMBEDDING_DIMENSION