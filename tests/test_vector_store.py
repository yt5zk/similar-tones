"""
VectorStoreのユニットテスト

ベクトル保存・読み込み・検索機能のテスト
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.vector_store import VectorStore


class TestVectorStore:
    """VectorStoreのテストクラス"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.vector_store = VectorStore()
        self.test_files = []  # 後でクリーンアップ用
    
    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ"""
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def _create_test_vectors(self, count: int = 5, dimension: int = 10) -> tuple:
        """テスト用ベクトルとファイルパスを生成"""
        # ランダムベクトル生成（再現性のためseed設定）
        np.random.seed(42)
        vectors = np.random.randn(count, dimension).astype(np.float32)
        
        # ファイルパスリスト
        file_paths = [f"/test/audio/file_{i}.wav" for i in range(count)]
        
        return vectors, file_paths
    
    def test_initialization(self):
        """VectorStoreの初期化テスト"""
        vector_store = VectorStore()
        assert vector_store is not None
    
    def test_save_index_success(self):
        """save_indexメソッドの正常動作テスト"""
        vectors, file_paths = self._create_test_vectors(3, 512)
        
        # 一時ファイルパス
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            output_path = Path(temp_file.name)
        
        self.test_files.append(str(output_path))
        
        # 保存実行
        self.vector_store.save_index(vectors, file_paths, output_path)
        
        # ファイルが作成されたか確認
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_save_index_mismatched_lengths(self):
        """save_indexでベクトル数とファイルパス数が一致しない場合のエラーテスト"""
        vectors, file_paths = self._create_test_vectors(3, 512)
        file_paths_short = file_paths[:2]  # 1つ少ない
        
        output_path = Path("/tmp/test_index.pkl")
        
        with pytest.raises(ValueError, match="Vectors count .* must match file paths count"):
            self.vector_store.save_index(vectors, file_paths_short, output_path)
    
    def test_save_index_empty_vectors(self):
        """save_indexで空のベクトル配列を渡した場合のエラーテスト"""
        empty_vectors = np.array([]).reshape(0, 512)
        empty_file_paths = []
        
        output_path = Path("/tmp/test_index.pkl")
        
        with pytest.raises(ValueError, match="Cannot save empty index"):
            self.vector_store.save_index(empty_vectors, empty_file_paths, output_path)
    
    def test_load_index_success(self):
        """load_indexメソッドの正常動作テスト"""
        vectors, file_paths = self._create_test_vectors(3, 512)
        
        # 一時ファイルパス
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            index_path = Path(temp_file.name)
        
        self.test_files.append(str(index_path))
        
        # 保存してから読み込み
        self.vector_store.save_index(vectors, file_paths, index_path)
        loaded_vectors, loaded_file_paths = self.vector_store.load_index(index_path)
        
        # 検証
        assert isinstance(loaded_vectors, np.ndarray)
        assert isinstance(loaded_file_paths, list)
        assert loaded_vectors.shape == vectors.shape
        assert loaded_file_paths == file_paths
        
        # 数値的な一致確認
        np.testing.assert_array_almost_equal(loaded_vectors, vectors)
    
    def test_load_index_file_not_found(self):
        """load_indexで存在しないファイルを指定した場合のエラーテスト"""
        non_existent_path = Path("/non/existent/index.pkl")
        
        with pytest.raises(FileNotFoundError, match="Index file not found"):
            self.vector_store.load_index(non_existent_path)
    
    def test_load_index_invalid_format(self):
        """load_indexで不正なファイル形式の場合のエラーテスト"""
        # 無効なファイルを作成
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            invalid_path = Path(temp_file.name)
            temp_file.write(b"invalid pickle data")
        
        self.test_files.append(str(invalid_path))
        
        with pytest.raises(RuntimeError, match="Index loading failed"):
            self.vector_store.load_index(invalid_path)
    
    def test_search_success(self):
        """searchメソッドの正常動作テスト"""
        vectors, file_paths = self._create_test_vectors(5, 10)
        
        # クエリベクトル（最初のベクトルと同じ）
        query_vector = vectors[0]
        
        # 検索実行
        results = self.vector_store.search(query_vector, vectors, file_paths, top_k=3)
        
        # 検証
        assert isinstance(results, list)
        assert len(results) == 3
        
        # 結果の構造確認
        for i, result in enumerate(results):
            assert isinstance(result, dict)
            assert "file_path" in result
            assert "similarity_score" in result
            assert "rank" in result
            assert result["rank"] == i + 1
            assert isinstance(result["similarity_score"], float)
        
        # 最初の結果が最も類似（クエリと同じベクトル）
        assert results[0]["similarity_score"] >= results[1]["similarity_score"]
        assert results[0]["file_path"] == file_paths[0]
        assert abs(results[0]["similarity_score"] - 1.0) < 1e-6  # コサイン類似度1.0に近い
    
    def test_search_empty_index(self):
        """searchで空のインデックスを指定した場合のテスト"""
        query_vector = np.array([1.0, 2.0, 3.0])
        empty_vectors = np.array([]).reshape(0, 3)
        empty_file_paths = []
        
        results = self.vector_store.search(query_vector, empty_vectors, empty_file_paths)
        
        assert results == []
    
    def test_search_mismatched_lengths(self):
        """searchでベクトル数とファイルパス数が一致しない場合のエラーテスト"""
        vectors, file_paths = self._create_test_vectors(3, 10)
        file_paths_short = file_paths[:2]  # 1つ少ない
        
        query_vector = vectors[0]
        
        with pytest.raises(ValueError, match="Index vectors count .* must match file paths count"):
            self.vector_store.search(query_vector, vectors, file_paths_short)
    
    def test_search_zero_query_vector(self):
        """searchでゼロベクトルをクエリに指定した場合のエラーテスト"""
        vectors, file_paths = self._create_test_vectors(3, 10)
        zero_query = np.zeros(10)
        
        with pytest.raises(ValueError, match="Query vector cannot be zero vector"):
            self.vector_store.search(zero_query, vectors, file_paths)
    
    def test_search_with_zero_index_vectors(self):
        """searchでインデックスにゼロベクトルが含まれる場合のテスト"""
        vectors, file_paths = self._create_test_vectors(3, 10)
        
        # 1つのベクトルをゼロベクトルに変更
        vectors[1] = np.zeros(10)
        
        query_vector = vectors[0]
        
        # 検索実行（ゼロベクトルは除外されるべき）
        results = self.vector_store.search(query_vector, vectors, file_paths, top_k=3)
        
        # ゼロベクトルが除外されて2つの結果が返されるはず
        assert len(results) == 2
        assert results[0]["file_path"] == file_paths[0]  # 自分自身
        assert results[1]["file_path"] == file_paths[2]  # 3番目のベクトル
    
    def test_search_all_zero_index_vectors(self):
        """searchでインデックスが全てゼロベクトルの場合のテスト"""
        vectors = np.zeros((3, 10))
        file_paths = [f"/test/file_{i}.wav" for i in range(3)]
        query_vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        results = self.vector_store.search(query_vector, vectors, file_paths)
        
        # 全てゼロベクトルなので結果は空
        assert results == []
    
    def test_search_similarity_ordering(self):
        """searchの類似度順序のテスト"""
        # 特定のベクトルを手動で作成
        vectors = np.array([
            [1.0, 0.0, 0.0],  # クエリと同じ方向
            [0.0, 1.0, 0.0],  # 直交
            [-1.0, 0.0, 0.0], # 逆方向
            [0.5, 0.5, 0.0],  # 45度
        ], dtype=np.float32)
        
        file_paths = [f"/test/file_{i}.wav" for i in range(4)]
        query_vector = np.array([1.0, 0.0, 0.0])
        
        results = self.vector_store.search(query_vector, vectors, file_paths, top_k=4)
        
        # 類似度順序の確認
        assert len(results) == 4
        
        # 1番目: 完全一致（コサイン類似度 1.0）
        assert results[0]["file_path"] == file_paths[0]
        assert abs(results[0]["similarity_score"] - 1.0) < 1e-6
        
        # 2番目: 45度（コサイン類似度 約0.707）
        assert results[1]["file_path"] == file_paths[3]
        assert abs(results[1]["similarity_score"] - (0.5 / np.sqrt(0.5))) < 1e-6
        
        # 3番目: 直交（コサイン類似度 0.0）
        assert results[2]["file_path"] == file_paths[1]
        assert abs(results[2]["similarity_score"] - 0.0) < 1e-6
        
        # 4番目: 逆方向（コサイン類似度 -1.0）
        assert results[3]["file_path"] == file_paths[2]
        assert abs(results[3]["similarity_score"] - (-1.0)) < 1e-6
    
    def test_search_top_k_limit(self):
        """searchのtop_k制限のテスト"""
        vectors, file_paths = self._create_test_vectors(10, 10)
        query_vector = vectors[0]
        
        # top_k=3で検索
        results = self.vector_store.search(query_vector, vectors, file_paths, top_k=3)
        
        assert len(results) == 3
        assert all(result["rank"] <= 3 for result in results)
        
        # top_k=15（インデックス数より多い）で検索
        results_all = self.vector_store.search(query_vector, vectors, file_paths, top_k=15)
        
        assert len(results_all) == 10  # インデックス数と同じ


class TestVectorStoreIntegration:
    """VectorStoreの統合テスト"""
    
    def test_save_load_search_pipeline(self):
        """保存→読み込み→検索の統合テスト"""
        vector_store = VectorStore()
        
        # テストデータ作成
        np.random.seed(42)
        vectors = np.random.randn(5, 100).astype(np.float32)
        file_paths = [f"/test/audio/preset_{i}.wav" for i in range(5)]
        
        # 一時ファイルパス
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            index_path = Path(temp_file.name)
        
        try:
            # 1. インデックス保存
            vector_store.save_index(vectors, file_paths, index_path)
            
            # 2. インデックス読み込み
            loaded_vectors, loaded_file_paths = vector_store.load_index(index_path)
            
            # 3. 検索実行
            query_vector = vectors[0]  # 最初のベクトルをクエリに使用
            results = vector_store.search(query_vector, loaded_vectors, loaded_file_paths, top_k=3)
            
            # 4. 結果検証
            assert len(results) == 3
            assert results[0]["file_path"] == file_paths[0]  # 自分自身が最も類似
            assert results[0]["similarity_score"] >= results[1]["similarity_score"]
            assert results[1]["similarity_score"] >= results[2]["similarity_score"]
            
        finally:
            # クリーンアップ
            if index_path.exists():
                index_path.unlink()