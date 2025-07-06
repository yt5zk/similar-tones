#!/usr/bin/env python3
"""
Similar Tones CLI Runner

CLIアプリケーションを実行するためのランナースクリプト
"""

import sys
from pathlib import Path

# srcディレクトリをパスに追加
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# main.pyから実行
from src.main import app

if __name__ == "__main__":
    app()