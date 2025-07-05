"""
ClapModelのユニットテスト

CLAP特徴抽出モデルの動作確認テスト
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from src.models.clap_model import ClapModel
from src.models.interface import EmbeddingModelInterface


class TestClapModel:
    """ClapModelのテストクラス"""
    
    @pytest.fixture
    def mock_clap_model(self):
        """モックされたClapModelを返すフィクスチャ"""
        with patch('src.models.clap_model.HFClapModel') as mock_hf_model, \
             patch('src.models.clap_model.ClapProcessor') as mock_processor:
            
            # モックの設定
            mock_model_instance = MagicMock()
            mock_processor_instance = MagicMock()
            
            mock_hf_model.from_pretrained.return_value = mock_model_instance
            mock_processor.from_pretrained.return_value = mock_processor_instance
            
            # プロセッサの設定
            mock_processor_instance.feature_extractor.sampling_rate = 48000
            mock_processor_instance.feature_extractor.max_length_s = 10
            
            # インスタンス作成
            clap_model = ClapModel(device="cpu")
            
            return clap_model, mock_model_instance, mock_processor_instance
    
    def test_inheritance(self):
        """EmbeddingModelInterfaceを正しく継承しているかテスト"""
        assert issubclass(ClapModel, EmbeddingModelInterface)
    
    def test_model_constants(self):
        """モデル定数の確認"""
        assert ClapModel.MODEL_NAME == "laion/clap-htsat-unfused"
        assert ClapModel.EMBEDDING_DIMENSION == 512
    
    def test_get_embedding_dimension(self, mock_clap_model):
        """get_embedding_dimensionメソッドのテスト"""
        clap_model, _, _ = mock_clap_model
        
        dimension = clap_model.get_embedding_dimension()
        assert dimension == 512
        assert isinstance(dimension, int)
    
    def test_device_auto_selection_cpu(self, mock_clap_model):
        """CPU環境でのデバイス自動選択テスト"""
        clap_model, mock_model, _ = mock_clap_model
        
        # CPUが選択されることを確認
        assert clap_model.device == "cpu"
        mock_model.to.assert_called_with("cpu")
        mock_model.eval.assert_called_once()
    
    def test_device_manual_setting(self):
        """手動デバイス設定のテスト"""
        with patch('src.models.clap_model.HFClapModel') as mock_hf_model, \
             patch('src.models.clap_model.ClapProcessor') as mock_processor:
            
            mock_model_instance = MagicMock()
            mock_processor_instance = MagicMock()
            
            mock_hf_model.from_pretrained.return_value = mock_model_instance
            mock_processor.from_pretrained.return_value = mock_processor_instance
            
            mock_processor_instance.feature_extractor.sampling_rate = 48000
            mock_processor_instance.feature_extractor.max_length_s = 10
            
            # 手動でデバイス指定
            clap_model = ClapModel(device="cuda")
            
            assert clap_model.device == "cuda"
            mock_model_instance.to.assert_called_with("cuda")
    
    def test_get_embedding_success(self, mock_clap_model):
        """get_embeddingメソッドの正常動作テスト"""
        clap_model, mock_model, mock_processor = mock_clap_model
        
        # テスト用音声データ（1秒、48kHz）
        audio_data = np.random.randn(48000).astype(np.float32)
        
        # モックの戻り値設定
        mock_processor.return_value = {
            'input_features': torch.randn(1, 1, 1001, 64)
        }
        
        # 512次元の埋め込みベクトルをモック
        mock_output = torch.randn(1, 512)
        mock_model.get_audio_features.return_value = mock_output
        
        # 実行
        embedding = clap_model.get_embedding(audio_data)
        
        # 検証
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert embedding.shape == (512,)
        
        # モックが正しく呼ばれたか確認
        mock_processor.assert_called_once_with(
            audios=audio_data,
            sampling_rate=48000,
            return_tensors="pt"
        )
        mock_model.get_audio_features.assert_called_once()
    
    def test_get_embedding_model_not_loaded(self):
        """モデル未ロード時のエラーテスト"""
        # モデルロードを失敗させる
        with patch('src.models.clap_model.HFClapModel') as mock_hf_model:
            mock_hf_model.from_pretrained.side_effect = Exception("Model load failed")
            
            with pytest.raises(RuntimeError, match="CLAP model loading failed"):
                ClapModel()
    
    def test_get_embedding_runtime_error(self, mock_clap_model):
        """推論時のランタイムエラーテスト"""
        clap_model, mock_model, mock_processor = mock_clap_model
        
        # プロセッサでエラーを発生させる
        mock_processor.side_effect = Exception("Processing failed")
        
        audio_data = np.random.randn(48000).astype(np.float32)
        
        with pytest.raises(RuntimeError, match="Embedding generation failed"):
            clap_model.get_embedding(audio_data)
    
    def test_get_embedding_with_different_audio_lengths(self, mock_clap_model):
        """異なる長さの音声データでのテスト"""
        clap_model, mock_model, mock_processor = mock_clap_model
        
        # 異なる長さの音声データ
        test_cases = [
            np.random.randn(24000).astype(np.float32),  # 0.5秒
            np.random.randn(48000).astype(np.float32),  # 1秒
            np.random.randn(240000).astype(np.float32), # 5秒
        ]
        
        for audio_data in test_cases:
            # モックの設定
            mock_processor.return_value = {
                'input_features': torch.randn(1, 1, 1001, 64)
            }
            mock_output = torch.randn(1, 512)
            mock_model.get_audio_features.return_value = mock_output
            
            # 実行
            embedding = clap_model.get_embedding(audio_data)
            
            # すべて512次元であることを確認
            assert embedding.shape == (512,)
            assert embedding.dtype == np.float32
    
    def test_get_embedding_batch_dimension_handling(self, mock_clap_model):
        """バッチ次元の処理テスト"""
        clap_model, mock_model, mock_processor = mock_clap_model
        
        audio_data = np.random.randn(48000).astype(np.float32)
        
        # モックの設定
        mock_processor.return_value = {
            'input_features': torch.randn(1, 1, 1001, 64)
        }
        
        # バッチサイズ1の埋め込みベクトル
        mock_output = torch.randn(1, 512)
        mock_model.get_audio_features.return_value = mock_output
        
        # 実行
        embedding = clap_model.get_embedding(audio_data)
        
        # バッチ次元が除去されて1次元配列になることを確認
        assert embedding.shape == (512,)
        assert len(embedding.shape) == 1
    
    def test_get_embedding_dimension_mismatch_warning(self, mock_clap_model):
        """埋め込み次元数不一致時の警告テスト"""
        clap_model, mock_model, mock_processor = mock_clap_model
        
        audio_data = np.random.randn(48000).astype(np.float32)
        
        # モックの設定
        mock_processor.return_value = {
            'input_features': torch.randn(1, 1, 1001, 64)
        }
        
        # 異なる次元数の埋め込みベクトル（1024次元）
        mock_output = torch.randn(1, 1024)
        mock_model.get_audio_features.return_value = mock_output
        
        # ログキャプチャでテスト
        with patch('src.models.clap_model.logger') as mock_logger:
            embedding = clap_model.get_embedding(audio_data)
            
            # 警告が出力されることを確認
            mock_logger.warning.assert_called_once()
            assert "Unexpected embedding dimension" in str(mock_logger.warning.call_args)
        
        # 結果は正しく返されることを確認
        assert embedding.shape == (1024,)


class TestClapModelIntegration:
    """ClapModelの統合テスト（実際のモデルロードなし）"""
    
    def test_interface_compliance(self):
        """インターフェース準拠性のテスト"""
        # ClapModelが必要なメソッドを実装していることを確認
        required_methods = ['get_embedding', 'get_embedding_dimension']
        
        for method_name in required_methods:
            assert hasattr(ClapModel, method_name)
            assert callable(getattr(ClapModel, method_name))
    
    def test_embedding_model_interface_abstract_methods(self):
        """EmbeddingModelInterfaceの抽象メソッドテスト"""
        # インターフェースを直接インスタンス化できないことを確認
        with pytest.raises(TypeError):
            EmbeddingModelInterface()