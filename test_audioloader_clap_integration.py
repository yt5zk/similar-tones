"""
AudioLoader + ClapModel 統合テスト

実際の音声ファイルでAudioLoaderとClapModelの連携動作を確認
"""

import sys
import logging
import numpy as np
from pathlib import Path

# srcディレクトリをパスに追加
sys.path.append('src')
from audio_loader import AudioLoader
from models.clap_model import ClapModel

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_audio() -> Path:
    """テスト用音声ファイルを作成"""
    import soundfile as sf
    import tempfile
    
    # テスト用サイン波（440Hz、3秒間、44.1kHzステレオ）
    sample_rate = 44100
    duration = 3.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    left_channel = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    right_channel = np.sin(2 * np.pi * frequency * 1.5 * t).astype(np.float32)  # 少し違う周波数
    
    # ステレオ音声として結合
    stereo_audio = np.column_stack([left_channel, right_channel])
    
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        file_path = Path(temp_file.name)
        sf.write(str(file_path), stereo_audio, sample_rate)
    
    logger.info(f"テスト音声ファイル作成: {file_path}")
    logger.info(f"形式: 44.1kHz ステレオ、長さ: {duration}秒")
    
    return file_path


def test_audioloader_clap_pipeline():
    """AudioLoader → ClapModel のパイプライン統合テスト"""
    logger.info("=== AudioLoader + ClapModel 統合テスト開始 ===")
    
    try:
        # 1. テスト音声ファイル作成
        test_audio_path = create_test_audio()
        
        # 2. AudioLoaderで音声処理
        logger.info("\\n--- AudioLoader処理 ---")
        audio_loader = AudioLoader()
        processed_audio = audio_loader.load(test_audio_path)
        
        logger.info(f"AudioLoader出力: {processed_audio.shape}, {processed_audio.dtype}")
        logger.info(f"音声長さ: {len(processed_audio) / 48000:.3f}秒")
        logger.info(f"データ範囲: [{processed_audio.min():.6f}, {processed_audio.max():.6f}]")
        
        # 3. ClapModelで特徴抽出
        logger.info("\\n--- ClapModel処理 ---")
        logger.info("CLAPモデルをロード中...")
        clap_model = ClapModel()
        
        logger.info("音声データから特徴ベクトルを生成中...")
        embedding = clap_model.get_embedding(processed_audio)
        
        logger.info(f"CLAP埋め込みベクトル: {embedding.shape}, {embedding.dtype}")
        logger.info(f"ベクトル次元: {clap_model.get_embedding_dimension()}")
        logger.info(f"ベクトル範囲: [{embedding.min():.6f}, {embedding.max():.6f}]")
        logger.info(f"ベクトルノルム: {np.linalg.norm(embedding):.6f}")
        
        # 4. 結果検証
        logger.info("\\n--- 結果検証 ---")
        
        # 形状確認
        assert embedding.shape == (512,), f"埋め込みベクトルの形状が不正: {embedding.shape}"
        assert embedding.dtype == np.float32, f"データ型が不正: {embedding.dtype}"
        
        # 値の妥当性確認
        assert not np.isnan(embedding).any(), "埋め込みベクトルにNaNが含まれています"
        assert not np.isinf(embedding).any(), "埋め込みベクトルに無限大が含まれています"
        assert np.linalg.norm(embedding) > 0, "埋め込みベクトルがゼロベクトルです"
        
        logger.info("✅ 全ての検証項目が合格しました")
        
        # 5. 同じ音声での再現性テスト
        logger.info("\\n--- 再現性テスト ---")
        embedding2 = clap_model.get_embedding(processed_audio)
        
        # 同じ音声から同じベクトルが生成されるか確認
        cosine_similarity = np.dot(embedding, embedding2) / (np.linalg.norm(embedding) * np.linalg.norm(embedding2))
        logger.info(f"同一音声のコサイン類似度: {cosine_similarity:.6f}")
        
        assert cosine_similarity > 0.999, f"再現性が低すぎます: {cosine_similarity}"
        logger.info("✅ 再現性テスト合格")
        
        return embedding
        
    except Exception as e:
        logger.error(f"統合テストでエラーが発生: {e}")
        raise
    
    finally:
        # クリーンアップ
        if 'test_audio_path' in locals():
            test_audio_path.unlink()
            logger.info(f"テストファイル削除: {test_audio_path}")


def test_different_audio_lengths():
    """異なる長さの音声での動作確認"""
    logger.info("\\n=== 異なる音声長での動作確認 ===")
    
    clap_model = ClapModel()
    audio_loader = AudioLoader()
    
    durations = [0.5, 1.0, 3.0, 5.0]  # 秒
    embeddings = []
    
    for duration in durations:
        logger.info(f"\\n--- {duration}秒音声テスト ---")
        
        # テスト音声生成
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # AudioLoaderで処理（ダミーファイルパスでテスト用に直接データを渡す）
        # 注意: 実際の実装では_convert_to_clap_formatを直接呼ぶ
        processed_audio = audio_loader._convert_to_clap_format(audio, sample_rate)
        
        # CLAP処理
        embedding = clap_model.get_embedding(processed_audio)
        embeddings.append(embedding)
        
        logger.info(f"音声長: {duration}秒 → 埋め込み: {embedding.shape}")
        logger.info(f"ベクトルノルム: {np.linalg.norm(embedding):.6f}")
    
    # 異なる長さの音声でも一貫したベクトルが生成されることを確認
    logger.info("\\n--- 長さ間の類似性確認 ---")
    for i, dur1 in enumerate(durations):
        for j, dur2 in enumerate(durations[i+1:], i+1):
            cosine_sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            logger.info(f"{dur1}秒 vs {dur2}秒: コサイン類似度 {cosine_sim:.6f}")


def main():
    """メイン実行関数"""
    try:
        # 基本統合テスト
        embedding = test_audioloader_clap_pipeline()
        
        # 異なる音声長でのテスト
        test_different_audio_lengths()
        
        logger.info("\\n=== 統合テスト完了 ===")
        logger.info("AudioLoader + ClapModelの連携が正常に動作しています")
        logger.info(f"最終埋め込みベクトル形状: {embedding.shape}")
        
    except Exception as e:
        logger.error(f"統合テスト失敗: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)