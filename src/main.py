"""
Similar Tones Application - Main Entry Point

CLIアプリケーションのメインエントリーポイント
"""

import typer
from typing import Optional
from pathlib import Path


app = typer.Typer(
    name="similar-tones",
    help="類似音色プリセット検索アプリケーション",
    add_completion=False
)


@app.command("index")
def create_index(
    preset_dir: Path = typer.Argument(
        ...,
        help="プリセット音源のディレクトリパス",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True
    ),
    output: Path = typer.Argument(
        ...,
        help="インデックスファイルの出力パス"
    )
):
    """
    プリセット音源からインデックスを作成します
    """
    typer.echo(f"インデックス作成開始: {preset_dir} -> {output}")
    # TODO: SearchService.create_index() を呼び出し
    typer.echo("インデックス作成機能は未実装です")


@app.command("search")
def search_similar(
    target: Path = typer.Argument(
        ...,
        help="ターゲット音源のパス",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    index: Path = typer.Argument(
        ...,
        help="インデックスファイルのパス",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        help="取得する類似プリセット数"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="CSV結果の出力パス（指定しない場合は標準出力）"
    )
):
    """
    類似音色プリセットを検索します
    """
    typer.echo(f"類似検索開始: {target}")
    typer.echo(f"インデックス: {index}")
    typer.echo(f"上位 {top_k} 件を取得")
    if output:
        typer.echo(f"結果出力先: {output}")
    else:
        typer.echo("結果出力先: 標準出力")
    
    # TODO: SearchService.find_similar() を呼び出し
    typer.echo("類似検索機能は未実装です")


if __name__ == "__main__":
    app()