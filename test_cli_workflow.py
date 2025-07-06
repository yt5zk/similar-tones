"""
CLI統合テスト

完全なCLIワークフローをテストする
"""

import sys
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_preset_directory() -> Path:
    """テスト用プリセットディレクトリを作成"""
    import soundfile as sf
    import numpy as np
    
    temp_dir = Path(tempfile.mkdtemp())
    logger.info(f"テストディレクトリ作成: {temp_dir}")
    
    # サブディレクトリ作成
    bass_dir = temp_dir / "bass"
    lead_dir = temp_dir / "lead"
    bass_dir.mkdir()
    lead_dir.mkdir()
    
    sample_rate = 44100
    duration = 2.0
    
    # Bass系音源作成
    for i in range(3):
        frequency = 80 + i * 20  # 80, 100, 120 Hz
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # エンベロープ適用
        attack_samples = int(0.1 * sample_rate)
        envelope = np.ones_like(audio)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        audio = audio * envelope * 0.7
        
        # ステレオ化
        stereo_audio = np.column_stack([audio, audio * 0.9])
        
        file_path = bass_dir / f"bass_{i:02d}.wav"
        sf.write(str(file_path), stereo_audio, sample_rate)
    
    # Lead系音源作成
    for i in range(2):
        frequency = 440 + i * 220  # 440, 660 Hz
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = (np.sin(2 * np.pi * frequency * t) + 
                0.3 * np.sin(2 * np.pi * frequency * 2 * t)).astype(np.float32)
        
        # エンベロープ適用
        audio = audio * 0.6
        
        # ステレオ化
        stereo_audio = np.column_stack([audio, audio])
        
        # OGGファイルとして保存
        from pydub import AudioSegment
        
        temp_wav = lead_dir / f"temp_{i}.wav"
        sf.write(str(temp_wav), stereo_audio, sample_rate)
        
        audio_segment = AudioSegment.from_wav(str(temp_wav))
        ogg_path = lead_dir / f"lead_{i:02d}.ogg"
        audio_segment.export(str(ogg_path), format="ogg")
        
        temp_wav.unlink()
    
    logger.info(f"Bass files: {len(list(bass_dir.glob('*.wav')))}")
    logger.info(f"Lead files: {len(list(lead_dir.glob('*.ogg')))}")
    
    return temp_dir


def create_target_audio() -> Path:
    """ターゲット音源を作成"""
    import soundfile as sf
    import numpy as np
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        target_path = Path(temp_file.name)
    
    # Bass系に似た音色を作成
    sample_rate = 44100
    duration = 1.5
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # 110Hzの低音
    audio = (np.sin(2 * np.pi * 110 * t) + 
            0.5 * np.sin(2 * np.pi * 110 * 2 * t)).astype(np.float32)
    
    # エンベロープ適用
    attack_samples = int(0.05 * sample_rate)
    envelope = np.ones_like(audio)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    audio = audio * envelope * 0.8
    
    # ステレオ化
    stereo_audio = np.column_stack([audio, audio * 0.95])
    sf.write(str(target_path), stereo_audio, sample_rate)
    
    logger.info(f"ターゲット音源作成: {target_path}")
    return target_path


def run_cli_command(args: list) -> tuple:
    """CLIコマンドを実行"""
    cmd = ["python", "run_cli.py"] + args
    logger.info(f"実行コマンド: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分タイムアウト
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logger.error("コマンドがタイムアウトしました")
        return -1, "", "Timeout"


def test_cli_index_command(preset_dir: Path) -> Path:
    """indexコマンドのテスト"""
    logger.info("=== CLI indexコマンドテスト ===")
    
    # インデックスファイルパス
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
        index_path = Path(temp_file.name)
    
    # indexコマンド実行
    returncode, stdout, stderr = run_cli_command([
        "index",
        str(preset_dir),
        str(index_path)
    ])
    
    logger.info(f"終了コード: {returncode}")
    logger.info(f"標準出力:\n{stdout}")
    if stderr:
        logger.warning(f"標準エラー:\n{stderr}")
    
    # 結果確認
    if returncode == 0:
        assert index_path.exists(), "インデックスファイルが作成されていません"
        assert index_path.stat().st_size > 0, "インデックスファイルが空です"
        logger.info("✅ indexコマンド成功")
        return index_path
    else:
        raise RuntimeError(f"indexコマンドが失敗しました: {stderr}")


def test_cli_search_command_console(target_path: Path, index_path: Path):
    """searchコマンド（コンソール出力）のテスト"""
    logger.info("=== CLI searchコマンド（コンソール出力）テスト ===")
    
    # searchコマンド実行（コンソール出力）
    returncode, stdout, stderr = run_cli_command([
        "search",
        str(target_path),
        str(index_path),
        "--top-k", "3"
    ])
    
    logger.info(f"終了コード: {returncode}")
    logger.info(f"標準出力:\n{stdout}")
    if stderr:
        logger.warning(f"標準エラー:\n{stderr}")
    
    # 結果確認
    if returncode == 0:
        assert "類似音色検索結果" in stdout, "検索結果ヘッダーが見つかりません"
        assert "bass" in stdout.lower() or "lead" in stdout.lower(), "音源ファイルが見つかりません"
        logger.info("✅ searchコマンド（コンソール出力）成功")
    else:
        raise RuntimeError(f"searchコマンドが失敗しました: {stderr}")


def test_cli_search_command_csv(target_path: Path, index_path: Path):
    """searchコマンド（CSV出力）のテスト"""
    logger.info("=== CLI searchコマンド（CSV出力）テスト ===")
    
    # CSV出力ファイルパス
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        csv_path = Path(temp_file.name)
    
    # searchコマンド実行（CSV出力）
    returncode, stdout, stderr = run_cli_command([
        "search",
        str(target_path),
        str(index_path),
        "--top-k", "5",
        "--output", str(csv_path)
    ])
    
    logger.info(f"終了コード: {returncode}")
    logger.info(f"標準出力:\n{stdout}")
    if stderr:
        logger.warning(f"標準エラー:\n{stderr}")
    
    # 結果確認
    if returncode == 0:
        assert csv_path.exists(), "CSVファイルが作成されていません"
        assert csv_path.stat().st_size > 0, "CSVファイルが空です"
        
        # CSV内容確認
        with open(csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"CSV内容:\n{content}")
        assert "rank,file_path,file_name,similarity_score" in content, "CSVヘッダーが正しくありません"
        
        lines = content.strip().split('\n')
        assert len(lines) >= 2, "データ行が見つかりません"  # ヘッダー + 少なくとも1データ行
        
        logger.info("✅ searchコマンド（CSV出力）成功")
        
        # CSVファイルクリーンアップ
        csv_path.unlink()
    else:
        raise RuntimeError(f"searchコマンド（CSV出力）が失敗しました: {stderr}")


def main():
    """メイン実行関数"""
    preset_dir = None
    target_path = None
    index_path = None
    
    try:
        logger.info("=== CLI統合テスト開始 ===")
        
        # 1. テスト用プリセットディレクトリ作成
        preset_dir = create_test_preset_directory()
        
        # 2. ターゲット音源作成
        target_path = create_target_audio()
        
        # 3. indexコマンドテスト
        index_path = test_cli_index_command(preset_dir)
        
        # 4. searchコマンド（コンソール出力）テスト
        test_cli_search_command_console(target_path, index_path)
        
        # 5. searchコマンド（CSV出力）テスト
        test_cli_search_command_csv(target_path, index_path)
        
        logger.info("=== CLI統合テスト完了 ===")
        logger.info("✅ 全てのCLIコマンドが正常に動作しています")
        
        return True
        
    except Exception as e:
        logger.error(f"CLI統合テスト失敗: {e}")
        return False
    
    finally:
        # クリーンアップ
        logger.info("=== クリーンアップ ===")
        
        if target_path and target_path.exists():
            target_path.unlink()
            logger.info(f"ターゲット音源削除: {target_path}")
        
        if index_path and index_path.exists():
            index_path.unlink()
            logger.info(f"インデックスファイル削除: {index_path}")
        
        if preset_dir and preset_dir.exists():
            shutil.rmtree(preset_dir)
            logger.info(f"プリセットディレクトリ削除: {preset_dir}")


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)