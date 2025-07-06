"""
Similar Tones Application - Main Entry Point

CLIアプリケーションのメインエントリーポイント
"""

import typer
from typing import Optional
from pathlib import Path
import logging
import sys

try:
    # パッケージとして実行される場合
    from .search_service import SearchService
    from .result_formatter import ResultFormatter
except ImportError:
    # 直接実行される場合
    import sys
    from pathlib import Path
    
    # srcディレクトリをパスに追加
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    from search_service import SearchService
    from result_formatter import ResultFormatter


app = typer.Typer(
    name="similar-tones",
    help="類似音色プリセット検索アプリケーション",
    add_completion=False
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    
    try:
        # SearchServiceでインデックス作成
        search_service = SearchService()
        search_service.create_index(preset_dir, output)
        
        typer.echo(f"✅ インデックス作成完了: {output}")
        
    except Exception as e:
        typer.echo(f"❌ インデックス作成エラー: {e}", err=True)
        raise typer.Exit(1)


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
    
    try:
        # SearchServiceで類似検索実行
        search_service = SearchService()
        results = search_service.find_similar(target, index, top_k=top_k)
        
        # ResultFormatterで結果整形
        formatter = ResultFormatter()
        
        if output:
            # CSV出力
            formatter.save_csv(results, output)
            typer.echo(f"✅ 検索結果をCSVに保存: {output}")
        else:
            # コンソール出力
            console_output = formatter.to_console(results)
            typer.echo(console_output)
        
    except Exception as e:
        typer.echo(f"❌ 類似検索エラー: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()