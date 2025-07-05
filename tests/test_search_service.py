"""
SearchServiceのユニットテスト

インデックス作成と類似検索機能のテスト
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.search_service import SearchService
from src.models.interface import EmbeddingModelInterface


class MockEmbeddingModel(EmbeddingModelInterface):
    """テスト用のモック埋め込みモデル"""
    
    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.call_count = 0
    
    def get_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """決定的な埋め込みベクトルを生成"""
        self.call_count += 1
        # 音声データの長さに基づいて決定的なベクトルを生成
        np.random.seed(len(audio_data) % 1000)
        return np.random.randn(self.dimension).astype(np.float32)
    
    def get_embedding_dimension(self) -> int:
        return self.dimension


class TestSearchService:
    """SearchServiceのテストクラス"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.mock_embedding_model = MockEmbeddingModel(dimension=10)
        self.search_service = SearchService(embedding_model=self.mock_embedding_model)
        self.test_files = []  # 後でクリーンアップ用
        self.test_dirs = []   # 後でクリーンアップ用
    
    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ"""
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        
        for dir_path in self.test_dirs:
            if os.path.exists(dir_path):
                import shutil
                shutil.rmtree(dir_path)
    
    def _create_test_directory_with_audio(self, count: int = 3) -> Path:
        """テスト用ディレクトリと音声ファイルを作成"""
        import soundfile as sf
        
        # 一時ディレクトリ作成
        temp_dir = Path(tempfile.mkdtemp())
        self.test_dirs.append(str(temp_dir))
        
        # サブディレクトリも作成（再帰的検索のテスト用）
        sub_dir = temp_dir / "subdir"
        sub_dir.mkdir()
        
        sample_rate = 44100
        duration = 0.1
        
        # メインディレクトリにWAVファイルを作成（半分）
        wav_count = count // 2 + count % 2  # 奇数の場合は多めに
        for i in range(wav_count):
            frequency = 440.0 * (i + 1)
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            file_path = temp_dir / f"audio_{i}.wav"
            sf.write(str(file_path), audio, sample_rate)
        
        # サブディレクトリにOGGファイルを作成（残り）
        ogg_count = count - wav_count
        for i in range(ogg_count):
            frequency = 880.0 * (i + 1)
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            file_path = sub_dir / f"sub_audio_{i}.ogg"
            # OGGファイルとして保存（AudioSegmentを使用）
            from pydub import AudioSegment
            
            # 一時WAVファイルを作成してOGGに変換
            temp_wav = temp_dir / f"temp_{i}.wav"
            sf.write(str(temp_wav), audio, sample_rate)
            
            audio_segment = AudioSegment.from_wav(str(temp_wav))
            audio_segment.export(str(file_path), format="ogg")
            
            # 一時WAVファイルを削除
            temp_wav.unlink()
        
        return temp_dir
    
    def test_initialization_default_model(self):
        """SearchServiceのデフォルトモデルでの初期化テスト"""
        with patch('src.search_service.ClapModel') as mock_clap:
            mock_clap_instance = Mock()
            mock_clap.return_value = mock_clap_instance
            
            service = SearchService()
            
            assert service.embedding_model == mock_clap_instance
            mock_clap.assert_called_once()
    
    def test_initialization_custom_model(self):
        """SearchServiceのカスタムモデルでの初期化テスト"""
        custom_model = MockEmbeddingModel()
        service = SearchService(embedding_model=custom_model)
        
        assert service.embedding_model == custom_model
    
    def test_find_audio_files_recursive(self):
        """_find_audio_filesメソッドの再帰的検索テスト"""
        test_dir = self._create_test_directory_with_audio(4)
        
        audio_files = self.search_service._find_audio_files(test_dir)
        
        # WAVとOGGファイルが見つかることを確認
        assert len(audio_files) == 4
        
        # ファイル拡張子の確認
        wav_files = [f for f in audio_files if f.suffix == '.wav']
        ogg_files = [f for f in audio_files if f.suffix == '.ogg']
        
        # 4ファイルの場合: wav_count = 4//2 + 4%2 = 2 + 0 = 2, ogg_count = 4 - 2 = 2
        assert len(wav_files) == 2  # メインディレクトリのWAVファイル
        assert len(ogg_files) == 2  # サブディレクトリのOGGファイル
        
        # ソートされていることを確認
        sorted_files = sorted(audio_files, key=lambda x: str(x))
        assert audio_files == sorted_files
    
    def test_find_audio_files_empty_directory(self):
        """_find_audio_filesメソッドの空ディレクトリテスト"""
        temp_dir = Path(tempfile.mkdtemp())
        self.test_dirs.append(str(temp_dir))
        
        audio_files = self.search_service._find_audio_files(temp_dir)
        
        assert audio_files == []
    
    def test_create_index_success(self):
        """create_indexメソッドの正常動作テスト"""
        # テストディレクトリ作成
        test_dir = self._create_test_directory_with_audio(3)
        
        # 出力ファイルパス
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            output_path = Path(temp_file.name)
        self.test_files.append(str(output_path))
        
        # インデックス作成実行
        self.search_service.create_index(test_dir, output_path)
        
        # ファイルが作成されたか確認
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # モック埋め込みモデルが呼ばれたか確認
        assert self.mock_embedding_model.call_count == 3
    
    def test_create_index_nonexistent_directory(self):
        """create_indexで存在しないディレクトリを指定した場合のエラーテスト"""
        nonexistent_dir = Path("/nonexistent/directory")
        output_path = Path("/tmp/test_index.pkl")
        
        with pytest.raises(FileNotFoundError, match="Preset directory not found"):
            self.search_service.create_index(nonexistent_dir, output_path)
    
    def test_create_index_not_directory(self):
        """create_indexでディレクトリでないパスを指定した場合のエラーテスト"""
        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_path = Path(temp_file.name)
        self.test_files.append(str(file_path))
        
        output_path = Path("/tmp/test_index.pkl")
        
        with pytest.raises(ValueError, match="Preset path is not a directory"):
            self.search_service.create_index(file_path, output_path)
    
    def test_create_index_no_audio_files(self):
        """create_indexで音声ファイルが存在しない場合のエラーテスト"""
        # 空のディレクトリを作成
        temp_dir = Path(tempfile.mkdtemp())
        self.test_dirs.append(str(temp_dir))
        
        output_path = Path("/tmp/test_index.pkl")
        
        with pytest.raises(ValueError, match="No audio files found"):
            self.search_service.create_index(temp_dir, output_path)
    
    def test_find_similar_success(self):
        """find_similarメソッドの正常動作テスト"""
        # 1. インデックス作成
        test_dir = self._create_test_directory_with_audio(3)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            index_path = Path(temp_file.name)
        self.test_files.append(str(index_path))
        
        self.search_service.create_index(test_dir, index_path)
        
        # 2. 検索実行
        audio_files = list(test_dir.glob("**/*.wav"))
        target_path = audio_files[0]  # 最初のファイルをターゲットに使用
        
        results = self.search_service.find_similar(target_path, index_path, top_k=2)
        
        # 3. 結果検証
        assert isinstance(results, list)
        assert len(results) == 2
        
        # 結果の構造確認
        for result in results:
            assert "file_path" in result
            assert "file_name" in result
            assert "similarity_score" in result
            assert "rank" in result
            assert isinstance(result["similarity_score"], float)
            assert isinstance(result["rank"], int)
        
        # 決定的なモックなので最初の結果のランクが1であることを確認
        assert results[0]["rank"] == 1
    
    def test_find_similar_nonexistent_target(self):
        """find_similarで存在しないターゲットファイルを指定した場合のエラーテスト"""
        nonexistent_target = Path("/nonexistent/audio.wav")
        index_path = Path("/tmp/index.pkl")
        
        with pytest.raises(FileNotFoundError, match="Target audio file not found"):
            self.search_service.find_similar(nonexistent_target, index_path)
    
    def test_find_similar_nonexistent_index(self):
        """find_similarで存在しないインデックスファイルを指定した場合のエラーテスト"""
        # 一時音声ファイル作成
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            target_path = Path(temp_file.name)
            # ダミー音声データ
            audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410)).astype(np.float32)
            sf.write(str(target_path), audio, 44100)
        self.test_files.append(str(target_path))
        
        nonexistent_index = Path("/nonexistent/index.pkl")
        
        with pytest.raises(FileNotFoundError, match="Index file not found"):
            self.search_service.find_similar(target_path, nonexistent_index)
    
    @patch('src.search_service.AudioLoader')
    def test_create_index_audio_loading_error(self, mock_audio_loader_class):
        """create_indexで音声読み込みエラーが発生した場合のテスト"""
        # AudioLoaderのモック設定
        mock_audio_loader = Mock()
        mock_audio_loader.load.side_effect = Exception("Audio loading failed")
        mock_audio_loader_class.return_value = mock_audio_loader
        
        # SearchServiceを再作成（モックを使用するため）
        service = SearchService(embedding_model=self.mock_embedding_model)
        
        # テストディレクトリ作成
        test_dir = self._create_test_directory_with_audio(2)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            output_path = Path(temp_file.name)
        self.test_files.append(str(output_path))
        
        # 全ファイルで読み込みエラーが発生するため、ValueError が発生するはず
        with pytest.raises(ValueError, match="No valid embeddings generated"):
            service.create_index(test_dir, output_path)
    
    @patch('src.search_service.AudioLoader')
    def test_create_index_partial_errors(self, mock_audio_loader_class):
        """create_indexで一部ファイルでエラーが発生した場合のテスト"""
        # AudioLoaderのモック設定（一部でエラー）
        mock_audio_loader = Mock()
        
        def mock_load_side_effect(file_path):
            if "audio_0" in str(file_path):
                raise Exception("Loading failed for audio_0")
            else:
                # 正常なダミーデータを返す
                return np.random.randn(4410).astype(np.float32)
        
        mock_audio_loader.load.side_effect = mock_load_side_effect
        mock_audio_loader_class.return_value = mock_audio_loader
        
        # SearchServiceを再作成
        service = SearchService(embedding_model=self.mock_embedding_model)
        
        # テストディレクトリ作成
        test_dir = self._create_test_directory_with_audio(3)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            output_path = Path(temp_file.name)
        self.test_files.append(str(output_path))
        
        # 一部エラーがあっても処理が続行されるはず
        service.create_index(test_dir, output_path)
        
        # ファイルが作成されることを確認
        assert output_path.exists()


class TestSearchServiceIntegration:
    """SearchServiceの統合テスト"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.test_files = []
        self.test_dirs = []
    
    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ"""
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        
        for dir_path in self.test_dirs:
            if os.path.exists(dir_path):
                import shutil
                shutil.rmtree(dir_path)
    
    def test_create_index_then_search_pipeline(self):
        """インデックス作成→検索の統合テスト"""
        # モック埋め込みモデルを使用
        mock_model = MockEmbeddingModel(dimension=50)
        search_service = SearchService(embedding_model=mock_model)
        
        # 1. テストディレクトリと音声ファイル作成
        import soundfile as sf
        temp_dir = Path(tempfile.mkdtemp())
        self.test_dirs.append(str(temp_dir))
        
        sample_rate = 44100
        duration = 0.1
        
        # 3つの異なる音声ファイルを作成
        for i in range(3):
            frequency = 220.0 * (2 ** i)  # 異なる周波数
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            file_path = temp_dir / f"preset_{i}.wav"
            sf.write(str(file_path), audio, sample_rate)
        
        # 2. インデックス作成
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            index_path = Path(temp_file.name)
        self.test_files.append(str(index_path))
        
        search_service.create_index(temp_dir, index_path)
        
        # 3. 検索実行
        target_file = temp_dir / "preset_0.wav"
        results = search_service.find_similar(target_file, index_path, top_k=3)
        
        # 4. 結果検証
        assert len(results) == 3
        
        # 決定的なモックなので結果が一定であることを確認
        # （必ずしも自分自身が最高類似度とは限らない）
        assert results[0]["rank"] == 1
        assert len(results) > 0
        
        # 類似度が降順になっていることを確認
        for i in range(len(results) - 1):
            assert results[i]["similarity_score"] >= results[i+1]["similarity_score"]
        
        # すべての結果にfile_nameが含まれていることを確認
        for result in results:
            assert result["file_name"] == Path(result["file_path"]).name