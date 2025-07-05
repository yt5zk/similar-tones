"""
VectorStore + CLAP 統合テスト

実際のCLAP埋め込みベクトルでVectorStoreの動作を確認
"""

import sys
import logging
import numpy as np
from pathlib import Path
import tempfile
import os

# srcディレクトリをパスに追加
sys.path.append('src')
from audio_loader import AudioLoader
from models.clap_model import ClapModel
from vector_store import VectorStore

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_audio_files(count: int = 5) -> list:
    """複数のテスト音声ファイルを作成"""
    import soundfile as sf
    
    audio_files = []
    sample_rate = 44100
    duration = 2.0
    
    for i in range(count):
        # 異なる周波数のサイン波を生成
        frequency = 220.0 * (2 ** (i / 12))  # 音程を変える
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # ステレオ音声（左右で少し異なる周波数）
        left_channel = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        right_channel = np.sin(2 * np.pi * frequency * 1.05 * t).astype(np.float32)
        stereo_audio = np.column_stack([left_channel, right_channel])
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=f'_test_{i}.wav', delete=False) as temp_file:
            file_path = Path(temp_file.name)
            sf.write(str(file_path), stereo_audio, sample_rate)
            audio_files.append(file_path)
    
    logger.info(f"Created {count} test audio files")
    return audio_files


def test_vectorstore_with_clap_embeddings():
    """VectorStore + CLAP埋め込みベクトルの統合テスト"""
    logger.info("=== VectorStore + CLAP統合テスト開始 ===")
    
    audio_files = []
    index_file = None
    
    try:
        # 1. テスト音声ファイル作成
        logger.info("\\n--- テスト音声ファイル作成 ---")
        audio_files = create_test_audio_files(5)
        
        # 2. AudioLoaderとClapModelの初期化
        logger.info("\\n--- モデル初期化 ---")
        audio_loader = AudioLoader()
        clap_model = ClapModel()
        vector_store = VectorStore()
        
        # 3. 各音声ファイルから埋め込みベクトルを生成
        logger.info("\\n--- 埋め込みベクトル生成 ---")
        embeddings = []
        file_paths = []
        
        for i, audio_file in enumerate(audio_files):
            logger.info(f"Processing file {i+1}/{len(audio_files)}: {audio_file.name}")
            
            # 音声読み込み・前処理
            audio_data = audio_loader.load(audio_file)
            
            # CLAP埋め込み生成
            embedding = clap_model.get_embedding(audio_data)
            embeddings.append(embedding)
            file_paths.append(str(audio_file))
            
            logger.info(f"  埋め込み: {embedding.shape}, ノルム: {np.linalg.norm(embedding):.6f}")
        
        # 4. 埋め込みベクトルを配列に変換
        embeddings_array = np.array(embeddings)
        logger.info(f"\\n全埋め込み配列: {embeddings_array.shape}")
        
        # 5. インデックス保存
        logger.info("\\n--- インデックス保存 ---")
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            index_file = Path(temp_file.name)
        
        vector_store.save_index(embeddings_array, file_paths, index_file)
        logger.info(f"インデックス保存完了: {index_file}")
        
        # 6. インデックス読み込み
        logger.info("\\n--- インデックス読み込み ---")
        loaded_embeddings, loaded_file_paths = vector_store.load_index(index_file)
        
        logger.info(f"読み込み完了: {loaded_embeddings.shape}, {len(loaded_file_paths)}件")
        
        # データ整合性確認
        np.testing.assert_array_almost_equal(embeddings_array, loaded_embeddings)
        assert file_paths == loaded_file_paths
        logger.info("✅ データ整合性確認完了")
        
        # 7. 類似検索テスト
        logger.info("\\n--- 類似検索テスト ---")
        
        # 最初の音声をクエリとして使用
        query_audio = audio_loader.load(audio_files[0])
        query_embedding = clap_model.get_embedding(query_audio)
        
        logger.info(f"クエリ埋め込み: {query_embedding.shape}")
        
        # 検索実行
        search_results = vector_store.search(
            query_embedding, 
            loaded_embeddings, 
            loaded_file_paths, 
            top_k=3
        )
        
        logger.info(f"検索結果: {len(search_results)}件")
        
        # 結果表示
        for i, result in enumerate(search_results):
            file_name = Path(result["file_path"]).name
            logger.info(f"  {i+1}位: {file_name} (類似度: {result['similarity_score']:.6f})")
        
        # 8. 結果検証
        logger.info("\\n--- 結果検証 ---")
        
        # 最初の結果が自分自身であることを確認
        assert search_results[0]["file_path"] == str(audio_files[0])
        assert search_results[0]["similarity_score"] >= 0.999  # ほぼ1.0
        
        # 類似度が降順になっていることを確認
        for i in range(len(search_results) - 1):
            assert search_results[i]["similarity_score"] >= search_results[i+1]["similarity_score"]
        
        # ランクが正しく設定されていることを確認
        for i, result in enumerate(search_results):
            assert result["rank"] == i + 1
        
        logger.info("✅ 全ての検証項目が合格しました")
        
        # 9. 異なるクエリでの検索テスト
        logger.info("\\n--- 異なるクエリでの検索テスト ---")
        
        for i in range(1, min(3, len(audio_files))):
            query_audio = audio_loader.load(audio_files[i])
            query_embedding = clap_model.get_embedding(query_audio)
            
            results = vector_store.search(query_embedding, loaded_embeddings, loaded_file_paths, top_k=3)
            
            # 最初の結果が自分自身であることを確認
            assert results[0]["file_path"] == str(audio_files[i])
            assert results[0]["similarity_score"] >= 0.999
            
            logger.info(f"クエリ{i+1}: 最高類似度 {results[0]['similarity_score']:.6f}")
        
        logger.info("✅ 異なるクエリでの検索テスト合格")
        
        return True
        
    except Exception as e:
        logger.error(f"統合テストでエラーが発生: {e}")
        raise
    
    finally:
        # クリーンアップ
        logger.info("\\n--- クリーンアップ ---")
        for audio_file in audio_files:
            if audio_file and audio_file.exists():
                audio_file.unlink()
                logger.info(f"音声ファイル削除: {audio_file.name}")
        
        if index_file and index_file.exists():
            index_file.unlink()
            logger.info(f"インデックスファイル削除: {index_file.name}")


def test_vectorstore_performance():
    """VectorStoreの性能テスト"""
    logger.info("\\n=== VectorStore性能テスト開始 ===")
    
    # 大量のダミー埋め込みベクトルを作成
    np.random.seed(42)
    large_embeddings = np.random.randn(1000, 512).astype(np.float32)
    large_file_paths = [f"/dummy/file_{i}.wav" for i in range(1000)]
    
    vector_store = VectorStore()
    
    # 検索クエリ
    query_embedding = np.random.randn(512).astype(np.float32)
    
    # 性能測定
    import time
    start_time = time.time()
    
    results = vector_store.search(query_embedding, large_embeddings, large_file_paths, top_k=10)
    
    end_time = time.time()
    search_time = end_time - start_time
    
    logger.info(f"検索対象: {len(large_embeddings)}件")
    logger.info(f"検索時間: {search_time:.4f}秒")
    logger.info(f"検索結果: {len(results)}件")
    logger.info(f"1件あたりの検索時間: {search_time/len(large_embeddings)*1000:.4f}ms")
    
    # 結果検証
    assert len(results) == 10
    assert all(isinstance(r["similarity_score"], float) for r in results)
    
    logger.info("✅ 性能テスト完了")


def main():
    """メイン実行関数"""
    try:
        # 基本統合テスト
        test_vectorstore_with_clap_embeddings()
        
        # 性能テスト
        test_vectorstore_performance()
        
        logger.info("\\n=== 全統合テスト完了 ===")
        logger.info("VectorStore + CLAPの連携が正常に動作しています")
        
    except Exception as e:
        logger.error(f"統合テスト失敗: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)