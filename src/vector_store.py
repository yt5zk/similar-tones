"""
VectorStore - ベクトル永続化と類似検索

埋め込みベクトルのファイル保存・読み込みと類似ベクトル検索
"""

from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
import pickle
import json


logger = logging.getLogger(__name__)


class VectorStore:
    """
    ベクトルの永続化と検索を担当するクラス
    """
    
    def __init__(self):
        """
        VectorStoreの初期化
        """
        logger.info("VectorStore initialized")
    
    def save_index(
        self, 
        vectors: np.ndarray, 
        file_paths: List[str], 
        output_path: Path
    ) -> None:
        """
        埋め込みベクトルとファイルパス情報をインデックスファイルに保存
        
        Args:
            vectors: 埋め込みベクトル配列 (N x D)
            file_paths: 対応するファイルパスのリスト
            output_path: インデックスファイルの出力パス
        """
        if len(vectors) != len(file_paths):
            raise ValueError(f"Vectors count ({len(vectors)}) must match file paths count ({len(file_paths)})")
        
        if len(vectors) == 0:
            raise ValueError("Cannot save empty index")
        
        logger.info(f"Saving index with {len(vectors)} vectors to {output_path}")
        
        # 1. インデックスデータを辞書形式で構成
        index_data = {
            "vectors": vectors,
            "file_paths": file_paths,
            "metadata": {
                "count": len(vectors),
                "dimension": vectors.shape[1] if len(vectors.shape) > 1 else vectors.shape[0],
                "dtype": str(vectors.dtype)
            }
        }
        
        # 2. 出力ディレクトリを作成
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 3. pickleファイルに保存
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(index_data, f)
            logger.info(f"Index saved successfully: {len(vectors)} vectors, {vectors.shape[1]}D")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise RuntimeError(f"Index saving failed: {e}")
    
    def load_index(self, index_path: Path) -> Tuple[np.ndarray, List[str]]:
        """
        インデックスファイルから埋め込みベクトルとファイルパス情報を読み込み
        
        Args:
            index_path: インデックスファイルのパス
            
        Returns:
            (ベクトル配列, ファイルパスリスト) のタプル
        """
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        logger.info(f"Loading index from {index_path}")
        
        try:
            # 1. pickleファイルから読み込み
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            # 2. データ構造を検証
            if not isinstance(index_data, dict):
                raise ValueError("Invalid index file format: not a dictionary")
            
            required_keys = {"vectors", "file_paths", "metadata"}
            if not all(key in index_data for key in required_keys):
                raise ValueError(f"Invalid index file format: missing required keys {required_keys}")
            
            vectors = index_data["vectors"]
            file_paths = index_data["file_paths"]
            metadata = index_data["metadata"]
            
            # 3. データ整合性チェック
            if not isinstance(vectors, np.ndarray):
                raise ValueError("Vectors must be a numpy array")
            
            if not isinstance(file_paths, list):
                raise ValueError("File paths must be a list")
            
            if len(vectors) != len(file_paths):
                raise ValueError(f"Vectors count ({len(vectors)}) must match file paths count ({len(file_paths)})")
            
            if len(vectors) != metadata.get("count", 0):
                raise ValueError("Metadata count mismatch")
            
            logger.info(f"Index loaded successfully: {len(vectors)} vectors, {vectors.shape[1]}D")
            
            return vectors, file_paths
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise RuntimeError(f"Index loading failed: {e}")
    
    def search(
        self, 
        query_vector: np.ndarray, 
        index_vectors: np.ndarray, 
        file_paths: List[str], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        クエリベクトルに類似するベクトルを検索
        
        Args:
            query_vector: クエリベクトル (1D)
            index_vectors: インデックスベクトル配列 (N x D)
            file_paths: 対応するファイルパスリスト
            top_k: 取得する上位件数
            
        Returns:
            類似度順の検索結果リスト
            [{"file_path": str, "similarity_score": float, "rank": int}, ...]
        """
        if len(index_vectors) != len(file_paths):
            raise ValueError(f"Index vectors count ({len(index_vectors)}) must match file paths count ({len(file_paths)})")
        
        if len(index_vectors) == 0:
            return []
        
        logger.info(f"Searching for top {top_k} similar vectors")
        
        # Phase 1: 総当たり（Brute-force）によるコサイン類似度計算
        
        # 1. クエリベクトルを正規化
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("Query vector cannot be zero vector")
        
        query_normalized = query_vector / query_norm
        
        # 2. インデックスベクトルを正規化
        index_norms = np.linalg.norm(index_vectors, axis=1)
        valid_indices = index_norms > 0  # ゼロベクトルを除外
        
        if not np.any(valid_indices):
            logger.warning("No valid vectors found in index (all zero vectors)")
            return []
        
        valid_vectors = index_vectors[valid_indices]
        valid_file_paths = [file_paths[i] for i in range(len(file_paths)) if valid_indices[i]]
        valid_norms = index_norms[valid_indices]
        
        index_normalized = valid_vectors / valid_norms[:, np.newaxis]
        
        # 3. コサイン類似度を計算（内積）
        similarities = np.dot(index_normalized, query_normalized)
        
        # 4. 類似度順にソート（降順）
        sorted_indices = np.argsort(similarities)[::-1]
        
        # 5. 上位top_k件を抽出
        top_k = min(top_k, len(sorted_indices))
        results = []
        
        for i in range(top_k):
            idx = sorted_indices[i]
            result = {
                "file_path": valid_file_paths[idx],
                "similarity_score": float(similarities[idx]),
                "rank": i + 1
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} similar vectors")
        
        return results