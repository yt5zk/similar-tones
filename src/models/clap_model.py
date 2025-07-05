"""
ClapModel - CLAP特徴抽出モデルの実装

EmbeddingModelInterfaceの具体実装
"""

import numpy as np
import logging
from typing import Optional
from .interface import EmbeddingModelInterface

import torch
from transformers import ClapModel as HFClapModel, ClapProcessor


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
        try:
            logger.info(f"Loading CLAP model: {self.MODEL_NAME}")
            
            # 1. モデルとプロセッサをロード
            self.model = HFClapModel.from_pretrained(self.MODEL_NAME)
            self.processor = ClapProcessor.from_pretrained(self.MODEL_NAME)
            
            # 2. デバイス設定
            if self.device is None:
                # 自動デバイス選択
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model.to(self.device)
            logger.info(f"CLAP model loaded on device: {self.device}")
            
            # 3. 評価モードに設定（推論のみ）
            self.model.eval()
            
            # 4. 設定確認
            logger.info(f"Audio sampling rate: {self.processor.feature_extractor.sampling_rate}Hz")
            logger.info(f"Max audio length: {self.processor.feature_extractor.max_length_s}s")
            
        except Exception as e:
            logger.error(f"Failed to load CLAP model: {e}")
            raise RuntimeError(f"CLAP model loading failed: {e}")
    
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
        
        try:
            # 1. プロセッサで音声データを前処理
            # sampling_rate=48000を明示的に指定（AudioLoaderの出力仕様）
            inputs = self.processor(
                audios=audio_data,
                sampling_rate=48000,
                return_tensors="pt"
            )
            
            # 2. デバイスに移動
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 3. モデルで推論実行（勾配計算無効化）
            with torch.no_grad():
                outputs = self.model.get_audio_features(**inputs)
            
            # 4. 埋め込みベクトルを抽出してNumPy配列に変換
            embedding = outputs.cpu().numpy()
            
            # バッチ次元を除去（1次元の埋め込みベクトルとして返却）
            if embedding.shape[0] == 1:
                embedding = embedding[0]
            
            logger.debug(f"Generated embedding shape: {embedding.shape}")
            
            # 埋め込み次元数を検証
            if embedding.shape[0] != self.EMBEDDING_DIMENSION:
                logger.warning(f"Unexpected embedding dimension: {embedding.shape[0]} (expected: {self.EMBEDDING_DIMENSION})")
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    def get_embedding_dimension(self) -> int:
        """
        CLAP埋め込みベクトルの次元数を取得
        
        Returns:
            512（CLAPの埋め込み次元）
        """
        return self.EMBEDDING_DIMENSION