# Similar Tones

類似音色プリセット検索アプリケーション

## 概要

特定のオーディオサンプルに音色が類似するシンセサイザーのプリセットを検索するCLIアプリケーションです。

## 開発環境

### 前提条件

- Docker
- Docker Compose

### セットアップ

```bash
# コンテナをビルドして起動
docker compose up -d --build

# コンテナに入る
docker compose exec app bash

# 依存関係をインストール（コンテナ内で）
poetry install
```

### テスト実行

```bash
# 環境疎通テスト
docker compose exec app poetry run pytest tests/test_environment.py
```

## 使用方法

（実装完了後に追記予定）

## 開発状況

現在Step 0（開発環境構築）を実施中です。