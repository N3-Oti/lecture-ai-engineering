import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import time
import great_expectations as gx

# 必要に応じてログレベルを設定（大量のログが出る場合）
# import logging
# logging.basicConfig(level=logging.ERROR)


class DataLoader:
    """データロードを行うクラス"""

    @staticmethod
    def load_titanic_data(path=None):
        """Titanicデータセットを読み込む"""
        if path:
            return pd.read_csv(path)
        else:
            # ローカルのファイル
            local_path = "data/Titanic.csv"
            if os.path.exists(local_path):
                return pd.read_csv(local_path)
            else:
                print(f"Error: Data file not found at {local_path}")
                return None  # ファイルが見つからなければNoneを返す

    @staticmethod
    def preprocess_titanic_data(data):
        """Titanicデータを前処理する"""
        if data is None:
            return None, None  # データがない場合はNoneを返す

        # 必要な特徴量を選択
        data = data.copy()

        # 不要な列を削除
        columns_to_drop = []
        for col in ["PassengerId", "Name", "Ticket", "Cabin"]:
            if col in data.columns:
                columns_to_drop.append(col)

        if columns_to_drop:
            data.drop(columns_to_drop, axis=1, inplace=True)

        # 目的変数とその他を分離
        if "Survived" in data.columns:
            y = data["Survived"]
            X = data.drop("Survived", axis=1)
            return X, y
        else:
            # Survivedカラムがない場合（例: 推論用データ）
            return data, None


class DataValidator:
    """データバリデーションを行うクラス"""

    @staticmethod
    def validate_titanic_data(data):
        """Titanicデータセットの検証"""
        # DataFrameに変換
        if not isinstance(data, pd.DataFrame):
            # エラーメッセージを ValidationResult ライクな形式で返す
            return False, {
                "success": False,
                "results": [
                    {
                        "success": False,
                        "exception_info": {
                            "raised_exception": True,
                            "exception_message": "入力データがpd.DataFrameではありません",
                        },
                    }
                ],
            }

        try:
            # EphemeralDataContext を取得 (新しいAPI)
            # project_root_dir=None は多くの場合必要
            context = gx.get_context(project_root_dir=None)

            # Pandas DataFrame をデータアセットとして追加 (新しいAPI)
            # このアセット名を使って validator を取得します
            data_asset_name = "titanic_data_asset"
            data_asset = context.add_pandas_dataframe_asset(name=data_asset_name, dataframe=data)

            # 期待値スイートの名前を定義 (今回はコード内で期待値を定義)
            expectation_suite_name = "titanic_data_suite"

            # 期待値スイートを作成または更新
            # add_or_update_expectation_suite はスイートオブジェクトを返す
            expectation_suite = context.add_or_update_expectation_suite(expectation_suite_name=expectation_suite_name)

            # Great Expectations の ExpectationConfiguration オブジェクトとして期待値を定義し、スイートに追加
            # 古いバージョンの expect_column_... 関数ではなく、core.ExpectationConfiguration を使う
            expectations_to_add = [
                # 必須カラムの存在確認 (Pandasで行うか、GEのexpect_table_columns_to_existなどを使う)
                # ここではPandasでの事前チェックは DataLoader.preprocess_titanic_data で行い、
                # その後のバリデーションではXに含まれるカラムを検証すると想定
                # Pclass: 1, 2, 3 のいずれか
                gx.core.ExpectationConfiguration(
                    expectation_type="expect_column_distinct_values_to_be_in_set",
                    kwargs={"column": "Pclass", "value_set": [1, 2, 3]},
                ),
                # Sex: male または female
                gx.core.ExpectationConfiguration(
                    expectation_type="expect_column_distinct_values_to_be_in_set",
                    kwargs={"column": "Sex", "value_set": ["male", "female"]},
                ),
                # Age: 0から100の範囲 (欠損値はimputerで埋まっているはずだが、念のためnot_null=Falseやallow_relative_errorなどを検討)
                # SimpleImputer はNaNを埋めるので、NaNは存在しないと仮定
                gx.core.ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={"column": "Age", "min_value": 0, "max_value": 100},
                ),
                # SibSp: 0以上の整数
                gx.core.ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_greater_than_or_equal_to",
                    kwargs={"column": "SibSp", "min_value": 0},
                ),
                # Parch: 0以上の整数
                gx.core.ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_greater_than_or_equal_to",
                    kwargs={"column": "Parch", "min_value": 0},
                ),
                # Fare: 0以上の数値
                gx.core.ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_greater_than_or_equal_to",
                    kwargs={"column": "Fare", "min_value": 0},
                ),
                # Embarked: C, Q, S または None (NaN) を許容
                # SimpleImputer(strategy="most_frequent")でNaNは埋まるはずなので、Noneは不要かもしれないが、データによってはありうる
                # 前処理後のXではNaNはmost_frequentで埋まっているはずなので、value_set=["C", "Q", "S"]で十分かもしれない
                # 元のコードの value_set=["C", "Q", "S", ""] は空文字列を許容しているが、imputerは空文字列を生成しない
                # ここでは imputer が NaN を埋めた後の状態を検証すると仮定し、C, Q, S のみを許容するのが適切か
                # 元のコードのロジックに合わせるため、空文字列も許容するならそのまま含める
                # NaNを明示的に許容する場合は None を value_set に含める
                gx.core.ExpectationConfiguration(
                    expectation_type="expect_column_distinct_values_to_be_in_set",
                    kwargs={
                        "column": "Embarked",
                        "value_set": ["C", "Q", "S", ""],
                        "catch_exception": True,
                    },  # catch_exception=True を追加しておくとエラー時に検証が止まらない
                ),
                # Embarked カラムに欠損値（NaN）がないことを確認したい場合は追加
                # SimpleImputer で埋められるはずなので、これが失敗するのは問題がある
                # gx.core.ExpectationConfiguration(
                #     expectation_type="expect_column_values_to_not_be_null",
                #     kwargs={"column": "Embarked"}
                # ),
            ]

            # スイートに期待値を追加
            # add_expectation_configurations は期待値のリストを受け取れる
            expectation_suite.add_expectation_configurations(expectations_to_add)

            # スイートを保存（EphemeralDataContextなのでファイルには保存されないが、メモリ上では更新される）
            context.save_expectation_suite(expectation_suite=expectation_suite)

            # バリデーターを取得 (新しいAPI)
            # validator を取得する際に、データアセット名とスイート名を指定
            validator = context.get_validator(
                datasource_names=[data_asset.datasource.name],  # データアセットが属するデータソース名
                asset_names=[data_asset.name],  # データアセット名
                expectation_suite_name=expectation_suite.name,  # 期待値スイート名
            )

            # 検証実行 (validate() は ValidationResult オブジェクトを返します)
            validation_result = validator.validate()

            # 検証結果オブジェクトの success 属性で全体の成功/失敗を確認
            is_successful = validation_result.success

            # 戻り値は success boolean と ValidationResult オブジェクト自体
            return is_successful, validation_result

        except Exception as e:
            print(f"Great Expectations検証エラー: {e}")
            # エラー発生時も ValidationResult ライクな構造で返す
            return False, {"success": False, "results": [], "error": str(e)}


class ModelTester:
    """モデルテストを行うクラス"""

    # ... (このクラスは元のコードから変更なし) ...

    @staticmethod
    def create_preprocessing_pipeline():
        """前処理パイプラインを作成"""
        numeric_features = ["Age", "Fare", "SibSp", "Parch"]
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_features = ["Pclass", "Sex", "Embarked"]
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",  # 指定されていない列は削除
        )
        return preprocessor

    @staticmethod
    def train_model(X_train, y_train, model_params=None):
        """モデルを学習する"""
        if model_params is None:
            model_params = {"n_estimators": 100, "random_state": 42}

        # 前処理パイプラインを作成
        preprocessor = ModelTester.create_preprocessing_pipeline()

        # モデル作成
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(**model_params)),
            ]
        )

        # 学習
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """モデルを評価する"""
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": accuracy, "inference_time": inference_time}

    @staticmethod
    def save_model(model, path="models/titanic_model.pkl"):
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"titanic_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        return model_path  # 保存したパスを返す

    @staticmethod
    def load_model(path="models/titanic_model.pkl"):
        """モデルを読み込む"""
        if not os.path.exists(path):
            print(f"Error: Model file not found at {path}")
            return None
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def compare_with_baseline(current_metrics, baseline_threshold=0.75):
        """ベースラインと比較する"""
        if current_metrics is None or "accuracy" not in current_metrics:
            print("Warning: current_metrics is invalid.")
            return False
        return current_metrics["accuracy"] >= baseline_threshold


# テスト関数（pytestで実行可能）
# ... (この部分も元のコードから変更なし、ただし DataValidator の戻り値変更に合わせて修正が必要になるかも) ...
# 今回は __main__ ブロックの修正にフォーカスします。


if __name__ == "__main__":
    # データロード
    data = DataLoader.load_titanic_data()
    if data is None:
        print("データファイルの読み込みに失敗しました。処理を終了します。")
        exit(1)

    # 前処理 (特徴量と目的変数に分割)
    X, y = DataLoader.preprocess_titanic_data(data)
    if X is None:
        print("データの前処理に失敗しました。処理を終了します。")
        exit(1)

    # データバリデーション
    # DataValidator.validate_titanic_data は success boolean と ValidationResult オブジェクトを返す
    # 注意: validate_titanic_data は X (前処理後のデータフレーム) を受け取る
    success, validation_result = DataValidator.validate_titanic_data(X)

    print(f"データ検証結果: {'成功' if success else '失敗'}")

    # 検証が失敗した場合、個々の期待値の結果を詳細表示
    if not success:
        print("\n--- 検証失敗の詳細 ---")
        # validation_result は ValidationResult オブジェクトまたはエラー辞書
        # try-exceptで捕捉したエラーがあるか確認
        if isinstance(validation_result, dict) and "error" in validation_result:
            print(f"検証エラー: {validation_result['error']}")
        elif hasattr(validation_result, "results"):  # ValidationResult オブジェクトの場合
            # validation_result.results は個々の ExpectationValidationResult オブジェクトのリスト
            for result in validation_result.results:
                # result は個々の ExpectationValidationResult オブジェクト
                # 属性アクセス (.success, .expectation_config.expectation_type, .result) を使用
                is_expectation_success = result.success
                expectation_type = result.expectation_config.expectation_type
                result_details = result.result  # 結果の詳細データ

                # 失敗した期待値のみ表示
                if not is_expectation_success:
                    print(f"  期待値タイプ: {expectation_type}")
                    print(f"  成功: {is_expectation_success}")
                    # 元のコードの「結果」に相当する詳細データを表示
                    print(f"  詳細結果: {result_details}")
                    # 例外情報があれば表示
                    if result.exception_info and result.exception_info.raised_exception:
                        print(f"  例外情報: {result.exception_info.exception_message}")
        else:
            # 予期しない形式の戻り値の場合
            print(f"不明な検証結果形式: {validation_result}")

        print("データ検証に失敗しました。処理を終了します。")
        exit(1)

    # モデルのトレーニングと評価 (検証が成功した場合のみ実行)
    # train_test_split には X と y の両方が必要
    if y is None:
        print("目的変数 'Survived' がデータに存在しないため、モデル学習をスキップします。")
        # 検証は通ったが学習はできないケース
        exit(0)  # 検証は成功したので終了コード0でも良いかもしれない

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # パラメータ設定
    model_params = {"n_estimators": 100, "random_state": 42}

    # モデルトレーニング
    model = ModelTester.train_model(X_train, y_train, model_params)
    metrics = ModelTester.evaluate_model(model, X_test, y_test)

    print(f"\n精度: {metrics['accuracy']:.4f}")
    print(f"推論時間: {metrics['inference_time']:.4f}秒")

    # モデル保存
    model_path = ModelTester.save_model(model)
    print(f"モデル保存先: {model_path}")

    # ベースラインとの比較
    baseline_threshold = 0.75  # ModelTesterのデフォルト値を使用
    baseline_ok = ModelTester.compare_with_baseline(metrics, baseline_threshold)
    print(f"ベースライン比較 ({baseline_threshold:.2f}): {'合格' if baseline_ok else '不合格'}")

    # Optionally: Load and test the saved model
    # loaded_model = ModelTester.load_model(model_path)
    # if loaded_model:
    #     loaded_metrics = ModelTester.evaluate_model(loaded_model, X_test, y_test)
    #     print(f"読み込みモデル精度: {loaded_metrics['accuracy']:.4f}")
