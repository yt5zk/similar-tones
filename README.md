# Similar Tones

類似音色プリセット検索アプリケーション

## 概要

特定のオーディオサンプルに音色が類似するシンセサイザーのプリセットを検索するCLIアプリケーションです。

CLAP（Contrastive Language-Audio Pre-training）モデルを使用して音声の埋め込みベクトルを生成し、コサイン類似度による検索を行います。

## 機能

- **インデックス作成**: プリセット音源ディレクトリから検索用インデックスを生成
- **類似検索**: ターゲット音源に類似するプリセットを検索
- **複数形式対応**: WAV/OGG形式の音声ファイルをサポート
- **出力形式**: コンソール表示またはCSV出力

## 開発環境

### 前提条件

- Docker
- Git

### セットアップ

```bash
# リポジトリをクローン
git clone <repository-url>
cd similar-tones

# Dockerイメージをビルド
docker build -t similar-tones-dev .
```

### テスト実行

```bash
# 全テストを実行
docker run --rm -v "$(pwd):/app" -w /app similar-tones-dev python -m pytest

# 特定のテストを実行
docker run --rm -v "$(pwd):/app" -w /app similar-tones-dev python -m pytest tests/test_audio_loader.py -v
```

## 使用方法

### 基本的な使用手順

#### 1. インデックス作成

```bash
# プリセットディレクトリからインデックスを作成
python run_cli.py index /path/to/preset/directory index.pkl
```

#### 2. 類似検索

```bash
# コンソール出力
python run_cli.py search target_audio.wav index.pkl --top-k 5

# CSV出力
python run_cli.py search target_audio.wav index.pkl --top-k 10 --output results.csv
```

### Docker経由での実行

```bash
# インデックス作成
docker run --rm \
  -v "/path/to/presets:/presets:ro" \
  -v "$(pwd):/app" -w /app \
  similar-tones-dev \
  python run_cli.py index /presets preset_index.pkl

# 類似検索
docker run --rm \
  -v "/path/to/audio:/audio:ro" \
  -v "$(pwd):/app" -w /app \
  similar-tones-dev \
  python run_cli.py search /audio/target.wav preset_index.pkl --top-k 5
```

### 大量データでの実行

```bash
# 長時間実行時（バックグラウンド + スリープ防止）
caffeinate -i nohup docker run --rm \
  -v "/path/to/presets:/presets:ro" \
  -v "$(pwd):/app" -w /app \
  similar-tones-dev \
  python run_cli.py index /presets large_index.pkl \
  > index_creation.log 2>&1 &

# 進捗確認
tail -f index_creation.log
grep "Processing" index_creation.log | wc -l
```

## コマンドリファレンス

### index コマンド

プリセット音源からインデックスファイルを作成します。

```bash
python run_cli.py index [OPTIONS] PRESET_DIR OUTPUT
```

**引数:**
- `PRESET_DIR`: プリセット音源のディレクトリパス
- `OUTPUT`: インデックスファイルの出力パス

### search コマンド

類似音色プリセットを検索します。

```bash
python run_cli.py search [OPTIONS] TARGET INDEX
```

**引数:**
- `TARGET`: ターゲット音源のパス
- `INDEX`: インデックスファイルのパス

**オプション:**
- `--top-k INTEGER`: 取得する類似プリセット数 (default: 10)
- `--output PATH`: CSV結果の出力パス（指定しない場合は標準出力）

## 技術仕様

### アーキテクチャ

- **AudioLoader**: 音声ファイル読み込み・前処理（48kHz mono変換）
- **ClapModel**: CLAP埋め込みベクトル生成（512次元）
- **VectorStore**: ベクトル永続化・類似検索（コサイン類似度）
- **SearchService**: 全コンポーネント統合
- **ResultFormatter**: 検索結果整形・出力

### サポート形式

- **音声形式**: WAV, OGG
- **サンプリングレート**: 任意（内部で48kHzに変換）
- **チャンネル**: モノラル・ステレオ（内部でモノラルに変換）

## 関連技術

- [CLAP: Contrastive Language-Audio Pre-training](https://github.com/LAION-AI/CLAP)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [librosa](https://librosa.org/)
- [Typer](https://typer.tiangolo.com/)