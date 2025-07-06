"""
ResultFormatter - 検索結果の整形と出力

検索結果をCSVやコンソール表示用に整形する
"""

from typing import List, Dict, Any
from pathlib import Path
import logging
import csv
import io


logger = logging.getLogger(__name__)


class ResultFormatter:
    """
    検索結果の整形と出力を担当するクラス
    """
    
    def __init__(self):
        """
        ResultFormatterの初期化
        """
        logger.info("ResultFormatter initialized")
    
    def to_csv(self, results: List[Dict[str, Any]]) -> str:
        """
        検索結果をCSV形式の文字列に変換
        
        Args:
            results: 検索結果リスト
                    [{"file_path": str, "similarity_score": float, "rank": int}, ...]
                    
        Returns:
            CSV形式の文字列（UTF-8）
        """
        logger.info(f"Formatting {len(results)} results to CSV")
        
        if not results:
            logger.warning("No results to format")
            return "rank,file_path,file_name,similarity_score\n"
        
        # 1. StringIOを使用してCSV文字列を生成
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 2. CSVヘッダーを書き込み
        writer.writerow(["rank", "file_path", "file_name", "similarity_score"])
        
        # 3. 結果データをCSV行として書き込み
        for result in results:
            writer.writerow([
                result["rank"],
                result["file_path"],
                result.get("file_name", Path(result["file_path"]).name),
                f"{result['similarity_score']:.6f}"
            ])
        
        # 4. CSV文字列として返却
        csv_string = output.getvalue()
        output.close()
        
        logger.debug(f"Generated CSV with {len(results)} rows")
        return csv_string
    
    def to_console(self, results: List[Dict[str, Any]]) -> str:
        """
        検索結果をコンソール表示用に整形
        
        Args:
            results: 検索結果リスト
                    
        Returns:
            コンソール表示用の文字列
        """
        logger.info(f"Formatting {len(results)} results for console display")
        
        if not results:
            return "検索結果が見つかりませんでした。\n"
        
        # 1. ヘッダー行を作成
        lines = []
        lines.append("=" * 80)
        lines.append(f"類似音色検索結果 ({len(results)}件)")
        lines.append("=" * 80)
        lines.append(f"{'Rank':<4} {'類似度':<8} {'ファイル名':<30} {'パス'}")
        lines.append("-" * 80)
        
        # 2. 結果データを整形
        for result in results:
            rank = result["rank"]
            similarity = result["similarity_score"]
            file_path = result["file_path"]
            file_name = result.get("file_name", Path(file_path).name)
            
            # ファイル名の長さ制限（30文字）
            if len(file_name) > 30:
                display_name = file_name[:27] + "..."
            else:
                display_name = file_name
            
            # パスの表示（相対パスまたは短縮）
            display_path = str(file_path)
            if len(display_path) > 40:
                display_path = "..." + display_path[-37:]
            
            lines.append(f"{rank:<4} {similarity:<8.3f} {display_name:<30} {display_path}")
        
        lines.append("=" * 80)
        
        console_output = "\n".join(lines) + "\n"
        logger.debug(f"Generated console output with {len(lines)} lines")
        
        return console_output
    
    def save_csv(self, results: List[Dict[str, Any]], output_path: Path) -> None:
        """
        検索結果をCSVファイルに保存
        
        Args:
            results: 検索結果リスト
            output_path: CSV出力ファイルのパス
        """
        logger.info(f"Saving {len(results)} results to CSV file: {output_path}")
        
        try:
            # 1. 出力ディレクトリを作成
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 2. CSV文字列を生成
            csv_content = self.to_csv(results)
            
            # 3. UTF-8エンコーディングでファイルに書き込み
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                f.write(csv_content)
            
            logger.info(f"CSV file saved successfully: {output_path}")
            
        except PermissionError as e:
            logger.error(f"Permission denied when saving CSV file: {e}")
            raise RuntimeError(f"CSV保存権限エラー: {output_path}")
        
        except Exception as e:
            logger.error(f"Failed to save CSV file: {e}")
            raise RuntimeError(f"CSV保存失敗: {e}")