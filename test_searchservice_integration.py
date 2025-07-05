"""
SearchService統合テスト

実際のCLAPモデルとSearchServiceの完全な動作確認
"""

import sys
import logging
import numpy as np
from pathlib import Path
import tempfile
import os

# srcディレクトリをパスに追加
sys.path.append('src')
from src.search_service import SearchService

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_preset_directory(count: int = 8) -> Path:
    """テスト用プリセットディレクトリを作成"""
    import soundfile as sf
    
    # 一時ディレクトリ作成
    temp_dir = Path(tempfile.mkdtemp())
    
    # サブディレクトリも作成（実際のプリセット構造をシミュレート）
    bass_dir = temp_dir / "bass"
    lead_dir = temp_dir / "lead"
    pad_dir = temp_dir / "pad"
    
    bass_dir.mkdir()
    lead_dir.mkdir() 
    pad_dir.mkdir()
    
    sample_rate = 44100
    duration = 3.0  # CLAPに適した長さ
    
    # 各カテゴリに異なる特性の音声を作成
    categories = [
        {"dir": bass_dir, "name": "bass", "freq_range": (80, 200), "count": count // 3},
        {"dir": lead_dir, "name": "lead", "freq_range": (440, 1760), "count": count // 3},
        {"dir": pad_dir, "name": "pad", "freq_range": (220, 880), "count": count - 2 * (count // 3)}
    ]
    
    file_index = 0
    for category in categories:
        for i in range(category["count"]):
            # 基本周波数を設定
            freq_min, freq_max = category["freq_range"]
            base_freq = freq_min + (freq_max - freq_min) * i / max(1, category["count"] - 1)
            
            # 時間軸
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # 複合波形を生成（より現実的な音色）
            audio = (
                np.sin(2 * np.pi * base_freq * t) +          # 基本波
                0.5 * np.sin(2 * np.pi * base_freq * 2 * t) + # 2倍音
                0.3 * np.sin(2 * np.pi * base_freq * 3 * t) + # 3倍音
                0.1 * np.random.randn(len(t))                  # ノイズ
            ).astype(np.float32)
            
            # 正規化
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            # エンベロープ適用（Attack-Decay-Sustain-Release）
            envelope = np.ones_like(audio)
            attack_samples = int(0.1 * sample_rate)
            release_samples = int(0.2 * sample_rate)
            
            # Attack
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            # Release
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)
            
            audio = audio * envelope
            
            # ステレオ化（左右で微妙に位相をずらす）
            left_channel = audio
            right_channel = audio * 0.9  # 右チャンネルを少し小さく
            stereo_audio = np.column_stack([left_channel, right_channel])
            
            # ファイル保存
            file_format = '.wav' if i % 2 == 0 else '.ogg'
            file_path = category["dir"] / f"{category['name']}_{i:02d}{file_format}"
            
            if file_format == '.wav':
                sf.write(str(file_path), stereo_audio, sample_rate)
            else:
                # OGG変換
                from pydub import AudioSegment
                temp_wav = category["dir"] / f"temp_{i}.wav"
                sf.write(str(temp_wav), stereo_audio, sample_rate)
                
                audio_segment = AudioSegment.from_wav(str(temp_wav))
                audio_segment.export(str(file_path), format="ogg")
                
                temp_wav.unlink()
            
            file_index += 1
    
    logger.info(f"Created test preset directory: {temp_dir}")
    logger.info(f"Bass presets: {len(list(bass_dir.glob('*')))} files")
    logger.info(f"Lead presets: {len(list(lead_dir.glob('*')))} files") 
    logger.info(f"Pad presets: {len(list(pad_dir.glob('*')))} files")
    
    return temp_dir


def test_searchservice_create_index():
    """SearchServiceのインデックス作成テスト"""
    logger.info("=== SearchService インデックス作成テスト ===")
    
    preset_dir = None
    index_file = None
    
    try:
        # 1. テストプリセットディレクトリ作成
        logger.info("\\n--- テストプリセットディレクトリ作成 ---")
        preset_dir = create_test_preset_directory(6)  # 6個のプリセット
        
        # 2. SearchService初期化
        logger.info("\\n--- SearchService初期化 ---")
        search_service = SearchService()  # デフォルトでClapModel使用
        
        # 3. インデックス作成
        logger.info("\\n--- インデックス作成 ---")
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            index_file = Path(temp_file.name)
        
        search_service.create_index(preset_dir, index_file)
        
        # 4. 結果検証
        logger.info("\\n--- 結果検証 ---")
        assert index_file.exists()
        assert index_file.stat().st_size > 0
        
        logger.info(f"✅ インデックス作成成功: {index_file}")
        logger.info(f"インデックスファイルサイズ: {index_file.stat().st_size} bytes")
        
        # 5. インデックス内容確認
        from src.vector_store import VectorStore
        vector_store = VectorStore()
        vectors, file_paths = vector_store.load_index(index_file)
        
        logger.info(f"インデックス内容: {len(vectors)} vectors, {vectors.shape[1]}D")
        logger.info(f"ファイルパス数: {len(file_paths)}")
        
        # 各カテゴリのファイルが含まれているか確認
        bass_files = [p for p in file_paths if 'bass' in p]
        lead_files = [p for p in file_paths if 'lead' in p]
        pad_files = [p for p in file_paths if 'pad' in p]
        
        logger.info(f"Bass: {len(bass_files)}件, Lead: {len(lead_files)}件, Pad: {len(pad_files)}件")
        
        assert len(bass_files) > 0
        assert len(lead_files) > 0
        assert len(pad_files) > 0
        
        return index_file, file_paths
        
    except Exception as e:
        logger.error(f"インデックス作成テストでエラー: {e}")
        raise
    
    finally:
        # クリーンアップはメイン関数で実行
        pass


def test_searchservice_find_similar(index_file: Path, indexed_file_paths: list):
    """SearchServiceの類似検索テスト"""
    logger.info("\\n=== SearchService 類似検索テスト ===")
    
    target_file = None
    
    try:
        # 1. ターゲット音声ファイル作成（bass系の音色）
        logger.info("\\n--- ターゲット音声ファイル作成 ---")
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            target_file = Path(temp_file.name)
        
        # bass系に似た音色を作成
        sample_rate = 44100
        duration = 2.5
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # 低音域の複合波形
        base_freq = 120  # bass範囲
        audio = (
            np.sin(2 * np.pi * base_freq * t) +
            0.4 * np.sin(2 * np.pi * base_freq * 2 * t) +
            0.2 * np.sin(2 * np.pi * base_freq * 3 * t)
        ).astype(np.float32)
        
        # エンベロープ適用
        attack_samples = int(0.05 * sample_rate)
        release_samples = int(0.1 * sample_rate)
        envelope = np.ones_like(audio)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        audio = audio * envelope
        
        # 正規化
        audio = audio / np.max(np.abs(audio)) * 0.7
        
        # ステレオ化
        stereo_audio = np.column_stack([audio, audio * 0.95])
        sf.write(str(target_file), stereo_audio, sample_rate)
        
        logger.info(f"ターゲットファイル作成: {target_file}")
        
        # 2. SearchService初期化
        search_service = SearchService()
        
        # 3. 類似検索実行
        logger.info("\\n--- 類似検索実行 ---")
        results = search_service.find_similar(target_file, index_file, top_k=5)
        
        # 4. 結果表示・検証
        logger.info("\\n--- 検索結果 ---")
        logger.info(f"検索結果数: {len(results)}")
        
        for i, result in enumerate(results):
            file_name = result["file_name"]
            similarity = result["similarity_score"]
            rank = result["rank"]
            
            # カテゴリ判定
            category = "unknown"
            if "bass" in result["file_path"]:
                category = "bass"
            elif "lead" in result["file_path"]:
                category = "lead"
            elif "pad" in result["file_path"]:
                category = "pad"
            
            logger.info(f"  {rank}位: {file_name} ({category}) - 類似度: {similarity:.6f}")
        
        # 5. 結果検証
        logger.info("\\n--- 結果検証 ---")
        
        # 基本的な構造チェック
        assert len(results) == 5
        assert all("file_path" in r for r in results)
        assert all("similarity_score" in r for r in results)
        assert all("rank" in r for r in results)
        
        # 類似度が降順になっているかチェック
        for i in range(len(results) - 1):
            assert results[i]["similarity_score"] >= results[i+1]["similarity_score"]
        
        # ランクが正しく設定されているかチェック
        for i, result in enumerate(results):
            assert result["rank"] == i + 1
        
        # bass系が上位に来ることを期待（必須ではないが傾向として）
        bass_results = [r for r in results if "bass" in r["file_path"]]
        if bass_results:
            logger.info(f"Bass系の結果: {len(bass_results)}件")
            highest_bass_rank = min(r["rank"] for r in bass_results)
            logger.info(f"最高位のBass: {highest_bass_rank}位")
        
        logger.info("✅ 類似検索テスト合格")
        
        return results
        
    except Exception as e:
        logger.error(f"類似検索テストでエラー: {e}")
        raise
    
    finally:
        # ターゲットファイルクリーンアップ
        if target_file and target_file.exists():
            target_file.unlink()
            logger.info(f"ターゲットファイル削除: {target_file}")


def test_searchservice_different_queries(index_file: Path):
    """異なるクエリでの検索テスト"""
    logger.info("\\n=== 異なるクエリでの検索テスト ===")
    
    target_files = []
    
    try:
        search_service = SearchService()
        
        # 3つの異なるカテゴリのクエリを作成
        queries = [
            {"name": "bass_query", "freq": 100, "category": "bass"},
            {"name": "lead_query", "freq": 880, "category": "lead"},
            {"name": "pad_query", "freq": 440, "category": "pad"}
        ]
        
        for query in queries:
            logger.info(f"\\n--- {query['name']} 作成・検索 ---")
            
            # クエリ音声作成
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                target_file = Path(temp_file.name)
                target_files.append(target_file)
            
            sample_rate = 44100
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # カテゴリに応じた音色生成
            if query["category"] == "bass":
                audio = np.sin(2 * np.pi * query["freq"] * t) + 0.3 * np.sin(2 * np.pi * query["freq"] * 2 * t)
            elif query["category"] == "lead":
                audio = np.sin(2 * np.pi * query["freq"] * t) + 0.2 * np.sin(2 * np.pi * query["freq"] * 3 * t)
            else:  # pad
                audio = (np.sin(2 * np.pi * query["freq"] * t) + 
                        np.sin(2 * np.pi * query["freq"] * 1.5 * t) +
                        np.sin(2 * np.pi * query["freq"] * 2 * t))
            
            audio = audio.astype(np.float32)
            audio = audio / np.max(np.abs(audio)) * 0.6
            
            # ステレオ化
            stereo_audio = np.column_stack([audio, audio])
            sf.write(str(target_file), stereo_audio, sample_rate)
            
            # 検索実行
            results = search_service.find_similar(target_file, index_file, top_k=3)
            
            # 結果表示
            logger.info(f"{query['name']} 検索結果:")
            for result in results:
                file_name = result["file_name"]
                similarity = result["similarity_score"]
                category = "unknown"
                if "bass" in result["file_path"]:
                    category = "bass"
                elif "lead" in result["file_path"]:
                    category = "lead" 
                elif "pad" in result["file_path"]:
                    category = "pad"
                
                logger.info(f"  {file_name} ({category}) - {similarity:.6f}")
        
        logger.info("✅ 異なるクエリでの検索テスト完了")
        
    except Exception as e:
        logger.error(f"異なるクエリ検索テストでエラー: {e}")
        raise
    
    finally:
        # クリーンアップ
        for target_file in target_files:
            if target_file.exists():
                target_file.unlink()


def main():
    """メイン実行関数"""
    preset_dir = None
    index_file = None
    
    try:
        # 1. インデックス作成テスト
        index_file, file_paths = test_searchservice_create_index()
        
        # 2. 類似検索テスト
        test_searchservice_find_similar(index_file, file_paths)
        
        # 3. 異なるクエリでの検索テスト
        test_searchservice_different_queries(index_file)
        
        logger.info("\\n=== 全統合テスト完了 ===")
        logger.info("SearchService (インデックス作成 + 類似検索) が正常に動作しています")
        
    except Exception as e:
        logger.error(f"統合テスト失敗: {e}")
        return False
    
    finally:
        # クリーンアップ
        logger.info("\\n--- クリーンアップ ---")
        
        if index_file and index_file.exists():
            index_file.unlink()
            logger.info(f"インデックスファイル削除: {index_file}")
        
        # プリセットディレクトリのクリーンアップは自動的に実行される（tempfile）
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)