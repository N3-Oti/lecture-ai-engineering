import os
import pytest
import pandas as pd
import numpy as np
import great_expectations as gx  # データ品質検証ライブラリ Great Expectations をインポート
from sklearn.datasets import fetch_openml  # scikit-learnからデータセットをダウンロードするために使用（現在は未使用）
import warnings

# 警告を抑制（Great Expectations が出す可能性のあるFutureWarningなどを非表示にする目的）
warnings.filterwarnings("ignore")

# テスト用データファイルのパスを定義します。
# このテストファイルが置かれているディレクトリからの相対パスで ../data/Titanic.csv を指します。
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")


@pytest.fixture  # pytestのフィクスチャとして定義。テスト関数でこの関数名を引数に指定すると、返り値が利用できる。
def sample_data():
    """
    Titanicテスト用データセットを読み込むフィクスチャ。
    DATA_PATH からCSVファイルを読み込み、DataFrameとして返す。
    """
    return pd.read_csv(DATA_PATH)


def test_data_exists(sample_data):  # sample_data フィクスチャを利用
    """
    【基本的なデータ存在チェック】
    読み込んだデータセットが空ではないこと、レコード（行）が存在することを確認するテスト。
    """
    assert not sample_data.empty, "データセットが空です"  # DataFrameが空でないか
    assert len(sample_data) > 0, "データセットにレコードがありません"  # DataFrameの行数 > 0 か


def test_data_columns(sample_data):
    """
    【カラム存在チェック】
    データセットに期待される必須のカラムが全て存在することを確認するテスト。
    機械学習モデルの学習や推論に必要な特徴量が欠けていないかを保証する。
    """
    # 期待されるカラム名のリスト
    expected_columns = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Survived",  # 目的変数も含む
    ]
    # 各期待カラムがデータセットのカラム一覧に含まれているかチェック
    for col in expected_columns:
        assert col in sample_data.columns, f"カラム '{col}' がデータセットに存在しません"


def test_data_types(sample_data):
    """
    【データ型チェック】
    各カラムのデータ型が期待通りであるか（数値型、カテゴリカル型など）を確認するテスト。
    データ型が異なると、前処理やモデル学習でエラーが発生する可能性があるため重要。
    """
    # 数値型であるべきカラムのリスト
    numeric_columns = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    for col in numeric_columns:
        # カラム内の非欠損値が数値型であるかを確認 (pd.api.types.is_numeric_dtype を使用)
        assert pd.api.types.is_numeric_dtype(
            sample_data[col].dropna()  # 欠損値を除外して型チェック
        ), f"カラム '{col}' が数値型ではありません"

    # カテゴリカル型（ここではobject型と仮定）であるべきカラムのリスト
    categorical_columns = ["Sex", "Embarked"]
    for col in categorical_columns:
        # カラムのdtypeが 'object' であるかを確認
        assert sample_data[col].dtype == "object", f"カラム '{col}' がカテゴリカル型ではありません"

    # 目的変数 'Survived' の値が期待通りか（0か1、または文字列の"0"か"1"）を確認
    survived_vals = sample_data["Survived"].dropna().unique()  # 欠損値を除外し、ユニークな値を取得
    # 値のセットが {0, 1} または {"0", "1"} の部分集合であることを確認
    assert set(survived_vals).issubset({"0", "1"}) or set(survived_vals).issubset({0, 1}), (
        "Survivedカラムには0, 1 (または文字列の'0', '1') のみ含まれるべきです"
    )


def test_missing_values_acceptable(sample_data):
    """
    【欠損値の許容範囲チェック】
    各カラムの欠損値の割合が、許容できる範囲内（ここでは80%未満）であるかを確認するテスト。
    欠損が多すぎると、その特徴量の信頼性が低下したり、モデル性能に悪影響を与えたりする可能性がある。
    """
    # 全てのカラムに対してループ
    for col in sample_data.columns:
        missing_rate = sample_data[col].isna().mean()  # isna()で欠損値True/Falseにし、mean()で欠損率を計算
        # 欠損率が80%未満であることをアサート
        assert missing_rate < 0.8, f"カラム '{col}' の欠損率が80%を超えています: {missing_rate:.2%}"


def test_value_ranges(sample_data):
    """
    【特定カラムの値範囲・値集合チェック (Great Expectationsを使用)】
    主要なカラムの値が、期待される範囲内にあるか、または期待される値の集合に含まれているかを確認するテスト。
    外れ値や不正なデータが含まれていないかを検証する。
    このテストでは、Great Expectations ライブラリを利用して、より宣言的に期待値を定義・検証している。
    """
    # Great Expectations のコンテキストを取得（または作成）
    context = gx.get_context()
    # Pandas DataFrame をデータソースとして追加
    data_source = context.data_sources.add_pandas("pandas_datasource")  # データソースに一意な名前を付ける
    # DataFrame をデータアセットとして登録
    data_asset = data_source.add_dataframe_asset(name="titanic_data_asset")  # アセットにも一意な名前

    # DataFrame全体を一つのバッチとして定義
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "titanic_batch_definition"  # バッチ定義にも一意な名前
    )
    # DataFrame を使ってバッチ（検証対象データ）を取得
    batch = batch_definition.get_batch(batch_parameters={"dataframe": sample_data})

    results = []  # Great Expectations の検証結果を格納するリスト

    # (本来は必須カラムチェックは test_data_columns でカバーされているが、GXでの例としても記述)
    # 必須カラムの存在確認 (Great Expectationsのテストとは直接関係ないが、前段として)
    required_columns = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ]
    missing_columns = [col for col in required_columns if col not in sample_data.columns]
    if missing_columns:
        # このテスト自体は失敗させず、警告を出力して早期リターンする例
        print(f"警告: Great Expectationsのテスト実行前に、以下の必須カラムがありません: {missing_columns}")
        # pytest.fail() を使えばテストを失敗させることも可能
        pytest.fail(f"Great Expectationsのテスト実行に必要なカラムがありません: {missing_columns}")

    # 検証したい期待値 (Expectation) のリストを定義
    expectations = [
        # 'Pclass' カラムの値は [1, 2, 3] のいずれかであること
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(column="Pclass", value_set=[1, 2, 3]),
        # 'Sex' カラムの値は ["male", "female"] のいずれかであること
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(column="Sex", value_set=["male", "female"]),
        # 'Age' カラムの値は 0 以上 100 以下であること (欠損値は評価対象外)
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Age",
            min_value=0,
            max_value=100,
            parse_strings_as_datetimes=False,  # 数値として比較
        ),
        # 'Fare' カラムの値は 0 以上 600 以下であること (欠損値は評価対象外)
        gx.expectations.ExpectColumnValuesToBeBetween(column="Fare", min_value=0, max_value=600),
        # 'Embarked' カラムの値は ["C", "Q", "S"] または欠損値（空文字列として扱う場合も考慮）のいずれかであること
        # 実際のデータに合わせて value_set に np.nan や None を含めるか、
        # GXの mostly パラメータで許容度を設定することも検討できる。
        # ここでは空文字列も許容する例として "" を追加。
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            column="Embarked",
            value_set=["C", "Q", "S", None, ""],  # 欠損をNoneや空文字列として許容する場合
        ),
    ]

    # 各期待値に対して検証を実行
    for expectation in expectations:
        result = batch.validate(expectation)  # バッチデータに対して期待値を検証
        results.append(result)  # 検証結果をリストに追加

    # 全ての検証結果が成功 (success=True) であることを確認
    is_successful = all(result.success for result in results)
    # 一つでも失敗があれば、アサーションエラーを発生させる
    assert is_successful, (
        f"Great Expectationsによるデータの値検証に失敗しました。詳細はログを確認してください。失敗した期待値: {[r.expectation_config.expectation_type for r in results if not r.success]}"
    )
