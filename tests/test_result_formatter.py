"""
ResultFormatterのユニットテスト

検索結果の整形機能のテスト
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.result_formatter import ResultFormatter


class TestResultFormatter:
    """ResultFormatterのテストクラス"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.formatter = ResultFormatter()
        self.test_files = []  # 後でクリーンアップ用
    
    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ"""
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def _create_sample_results(self) -> list:
        """テスト用検索結果データを作成"""
        return [
            {
                "rank": 1,
                "file_path": "/path/to/presets/bass/deep_bass_01.wav",
                "file_name": "deep_bass_01.wav",
                "similarity_score": 0.98765
            },
            {
                "rank": 2,
                "file_path": "/path/to/presets/lead/bright_lead.ogg",
                "file_name": "bright_lead.ogg",
                "similarity_score": 0.87432
            },
            {
                "rank": 3,
                "file_path": "/very/long/path/to/some/preset/directory/pad/ambient_pad_with_very_long_filename.wav",
                "file_name": "ambient_pad_with_very_long_filename.wav",
                "similarity_score": 0.76123
            }
        ]
    
    def test_initialization(self):
        """ResultFormatterの初期化テスト"""
        formatter = ResultFormatter()
        assert formatter is not None
    
    def test_to_csv_success(self):
        """to_csvメソッドの正常動作テスト"""
        results = self._create_sample_results()
        
        csv_output = self.formatter.to_csv(results)
        
        # CSVヘッダーの確認
        lines = csv_output.strip().split('\n')
        assert len(lines) == 4  # ヘッダー + 3データ行
        
        header = lines[0]
        assert "rank" in header
        assert "file_path" in header
        assert "file_name" in header
        assert "similarity_score" in header
        
        # データ行の確認
        first_data_line = lines[1]
        assert "1," in first_data_line
        assert "deep_bass_01.wav" in first_data_line
        assert "0.987650" in first_data_line  # 6桁精度
    
    def test_to_csv_empty_results(self):
        """to_csvメソッドの空結果テスト"""
        empty_results = []
        
        csv_output = self.formatter.to_csv(empty_results)
        
        # ヘッダーのみの確認
        lines = csv_output.strip().split('\n')
        assert len(lines) == 1
        assert "rank,file_path,file_name,similarity_score" in lines[0]
    
    def test_to_console_success(self):
        """to_consoleメソッドの正常動作テスト"""
        results = self._create_sample_results()
        
        console_output = self.formatter.to_console(results)
        
        # 基本構造の確認
        lines = console_output.split('\n')
        assert len(lines) > 5  # ヘッダー + データ行 + フッター
        
        # ヘッダーの確認
        assert "類似音色検索結果" in console_output
        assert "(3件)" in console_output
        
        # データの確認
        assert "deep_bass_01.wav" in console_output
        assert "0.988" in console_output  # 3桁精度
        assert "bright_lead.ogg" in console_output
        
        # 長いファイル名の切り詰めテスト
        assert "ambient_pad_with_very_long_..." in console_output
    
    def test_to_console_empty_results(self):
        """to_consoleメソッドの空結果テスト"""
        empty_results = []
        
        console_output = self.formatter.to_console(empty_results)
        
        assert "検索結果が見つかりませんでした" in console_output
    
    def test_save_csv_success(self):
        """save_csvメソッドの正常動作テスト"""
        results = self._create_sample_results()
        
        # 一時ファイルパス
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            output_path = Path(temp_file.name)
        self.test_files.append(str(output_path))
        
        # CSV保存実行
        self.formatter.save_csv(results, output_path)
        
        # ファイルが作成されたか確認
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # ファイル内容の確認
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "rank,file_path,file_name,similarity_score" in content
        assert "deep_bass_01.wav" in content
        assert "0.987650" in content
    
    def test_save_csv_create_directory(self):
        """save_csvメソッドのディレクトリ作成テスト"""
        results = self._create_sample_results()
        
        # 存在しないディレクトリのパス
        temp_dir = Path(tempfile.mkdtemp())
        output_path = temp_dir / "subdir" / "results.csv"
        self.test_files.append(str(output_path))
        
        # CSV保存実行（ディレクトリが自動作成されるはず）
        self.formatter.save_csv(results, output_path)
        
        # ファイルが作成されたか確認
        assert output_path.exists()
        assert output_path.parent.exists()
    
    def test_to_csv_file_name_fallback(self):
        """to_csvメソッドのfile_name不足時のフォールバックテスト"""
        # file_nameが含まれていない結果データ
        results_without_filename = [
            {
                "rank": 1,
                "file_path": "/path/to/test.wav",
                "similarity_score": 0.95
            }
        ]
        
        csv_output = self.formatter.to_csv(results_without_filename)
        
        # Pathからファイル名が自動抽出されることを確認
        assert "test.wav" in csv_output
    
    def test_to_console_path_truncation(self):
        """to_consoleメソッドのパス切り詰めテスト"""
        # 非常に長いパスを持つ結果データ
        long_path_results = [
            {
                "rank": 1,
                "file_path": "/very/very/very/long/path/to/some/deep/directory/structure/that/exceeds/normal/length/test.wav",
                "file_name": "test.wav",
                "similarity_score": 0.95
            }
        ]
        
        console_output = self.formatter.to_console(long_path_results)
        
        # パスが切り詰められていることを確認
        assert "..." in console_output
        lines = console_output.split('\n')
        data_line = next(line for line in lines if "test.wav" in line)
        
        # 行の長さが適切に制限されていることを確認
        assert len(data_line) <= 120  # 適切な長さ制限


class TestResultFormatterIntegration:
    """ResultFormatterの統合テスト"""
    
    def test_csv_console_consistency(self):
        """CSVとコンソール出力の一貫性テスト"""
        results = [
            {
                "rank": 1,
                "file_path": "/test/file1.wav",
                "file_name": "file1.wav",
                "similarity_score": 0.9876
            },
            {
                "rank": 2,
                "file_path": "/test/file2.ogg",
                "file_name": "file2.ogg",
                "similarity_score": 0.8765
            }
        ]
        
        formatter = ResultFormatter()
        
        # CSV出力
        csv_output = formatter.to_csv(results)
        
        # コンソール出力
        console_output = formatter.to_console(results)
        
        # 基本データの一貫性確認
        assert "file1.wav" in csv_output
        assert "file1.wav" in console_output
        assert "file2.ogg" in csv_output
        assert "file2.ogg" in console_output
        
        # ランク順序の一貫性確認
        csv_lines = csv_output.strip().split('\n')[1:]  # ヘッダー除去
        assert csv_lines[0].startswith('1,')  # 1位
        assert csv_lines[1].startswith('2,')  # 2位
    
    def test_round_trip_save_load(self):
        """保存→読み込みの整合性テスト"""
        import csv
        import io
        
        results = [
            {
                "rank": 1,
                "file_path": "/path/to/test.wav",
                "file_name": "test.wav",
                "similarity_score": 0.123456
            }
        ]
        
        formatter = ResultFormatter()
        
        # CSV文字列生成
        csv_content = formatter.to_csv(results)
        
        # CSV文字列をパース
        reader = csv.DictReader(io.StringIO(csv_content))
        parsed_results = list(reader)
        
        # データの整合性確認
        assert len(parsed_results) == 1
        parsed_row = parsed_results[0]
        
        assert parsed_row['rank'] == '1'
        assert parsed_row['file_path'] == '/path/to/test.wav'
        assert parsed_row['file_name'] == 'test.wav'
        assert float(parsed_row['similarity_score']) == pytest.approx(0.123456, abs=1e-6)