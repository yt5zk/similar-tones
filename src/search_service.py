"""
SearchService - アプリケーションの中核ロジック

インデックス作成と類似検索のユースケースを制御する
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import glob

from .audio_loader import AudioLoader
from .models.interface import EmbeddingModelInterface
from .models.clap_model import ClapModel
from .vector_store import VectorStore


logger = logging.getLogger(__name__)


class SearchService:
    """
    類似音色検索サービスのメインクラス
    """
    
    def __init__(self, embedding_model: Optional[EmbeddingModelInterface] = None):
        """
        SearchServiceの初期化
        
        Args:
            embedding_model: 埋め込みモデル（None の場合は ClapModel を使用）
        """
        self.audio_loader = AudioLoader()
        self.embedding_model = embedding_model or ClapModel()
        self.vector_store = VectorStore()
        
        logger.info("SearchService initialized with embedding model: %s", 
                   type(self.embedding_model).__name__)
    
    def create_index(self, preset_dir: Path, output_path: Path) -> None:
        """
        プリセットディレクトリからインデックスを作成
        
        Args:
            preset_dir: プリセット音源のディレクトリ
            output_path: インデックスファイルの出力パス
        """
        if not preset_dir.exists():
            raise FileNotFoundError(f"Preset directory not found: {preset_dir}")
        
        if not preset_dir.is_dir():
            raise ValueError(f"Preset path is not a directory: {preset_dir}")
        
        logger.info(f"Creating index from {preset_dir} -> {output_path}")
        
        # 1. preset_dir内の全音声ファイルを取得
        audio_files = self._find_audio_files(preset_dir)
        
        if not audio_files:
            raise ValueError(f"No audio files found in {preset_dir}")
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # 2. 各ファイルを処理して埋め込みベクトルを生成
        embeddings = []
        file_paths = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                logger.info(f"Processing {i+1}/{len(audio_files)}: {audio_file.name}")
                
                # AudioLoaderで音声を読み込み
                audio_data = self.audio_loader.load(audio_file)
                
                # EmbeddingModelで特徴量を計算
                embedding = self.embedding_model.get_embedding(audio_data)
                
                embeddings.append(embedding)
                file_paths.append(str(audio_file))
                
            except Exception as e:
                logger.warning(f"Failed to process {audio_file}: {e}")
                continue
        
        if not embeddings:
            raise ValueError("No valid embeddings generated")
        
        # 3. 埋め込みベクトルを配列に変換
        import numpy as np
        embeddings_array = np.array(embeddings)
        
        logger.info(f"Generated {len(embeddings_array)} embeddings with shape {embeddings_array.shape}")
        
        # 4. VectorStoreでインデックスファイルに保存
        self.vector_store.save_index(embeddings_array, file_paths, output_path)
        
        logger.info(f"Index creation completed: {len(embeddings_array)} files indexed")
    
    def find_similar(
        self, 
        target_path: Path, 
        index_path: Path, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        類似プリセットを検索
        
        Args:
            target_path: ターゲット音源のパス
            index_path: インデックスファイルのパス
            top_k: 取得する上位件数
            
        Returns:
            類似度順の検索結果リスト
        """
        if not target_path.exists():
            raise FileNotFoundError(f"Target audio file not found: {target_path}")
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        logger.info(f"Searching similar presets for {target_path}")
        logger.info(f"Using index: {index_path}, top_k: {top_k}")
        
        # 1. VectorStoreでインデックスをロード
        vectors, file_paths = self.vector_store.load_index(index_path)
        logger.info(f"Loaded index: {len(vectors)} vectors")
        
        # 2. AudioLoaderでターゲット音源を読み込み
        target_audio = self.audio_loader.load(target_path)
        logger.info(f"Loaded target audio: {target_audio.shape}")
        
        # 3. EmbeddingModelで特徴量を計算
        target_embedding = self.embedding_model.get_embedding(target_audio)
        logger.info(f"Generated target embedding: {target_embedding.shape}")
        
        # 4. VectorStoreで類似検索を実行
        search_results = self.vector_store.search(
            target_embedding, vectors, file_paths, top_k=top_k
        )
        
        # 5. 結果を整形して返却
        formatted_results = []
        for result in search_results:
            # ファイルパスをPathオブジェクトに変換して詳細情報を取得
            file_path = Path(result["file_path"])
            
            formatted_result = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "similarity_score": result["similarity_score"],
                "rank": result["rank"]
            }
            formatted_results.append(formatted_result)
        
        logger.info(f"Found {len(formatted_results)} similar presets")
        
        return formatted_results
    
    def _find_audio_files(self, directory: Path) -> List[Path]:
        """
        ディレクトリ内の音声ファイルを再帰的に検索
        
        Args:
            directory: 検索対象ディレクトリ
            
        Returns:
            音声ファイルのパスリスト
        """
        supported_extensions = ['.wav', '.ogg']
        audio_files = []
        
        for ext in supported_extensions:
            # 再帰的に検索（**/*pattern）
            pattern = f"**/*{ext}"
            files = list(directory.glob(pattern))
            audio_files.extend(files)
        
        # ファイル名でソート（再現性確保）
        audio_files.sort(key=lambda x: str(x))
        
        logger.info(f"Found {len(audio_files)} audio files with extensions {supported_extensions}")
        
        return audio_files