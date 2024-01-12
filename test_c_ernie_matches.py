from arfsk import AutonSKRFWrapper
from sklearn.ensemble import RandomForestClassifier
import unittest
import os
import shutil
import numpy as np
import pandas as pd


class TestCAutonRFMatches(unittest.TestCase):
    """Test to confirm the outputs of ARSKWrapper methods match the corresponding functionality in the C version of
        AutonRF.

        This will store temporary files in the location defined by tmp_dir

    """
    base_dir = os.path.join('test_data')
    tmp_dir = os.path.join(base_dir, 'tmp_legacy_data')

    train_file = os.path.join(tmp_dir, 'train.csv')
    test_file = os.path.join(tmp_dir, 'test.csv')

    model_file = os.path.join(tmp_dir, 'model.txt')
    pred_prefix = os.path.join(tmp_dir, 'test_')

    preds_suffix = 'predictions.csv'
    metrics_suffix = 'metrics.csv'
    attr_suffix = 'attributes.csv'

    fold_column = "FoldID"
    label_col = "LABEL"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Read in C_ERNIE environment variable. This is the path to the llnl_canes_ml_core executable
        try:
            cls._c_ernie = os.environ["C_ERNIE"]
        except KeyError:
            raise RuntimeError("You must set the C_ERNIE environment variable to the auton RF C library executable")

        # Create tmp dir, rm if necessary
        shutil.rmtree(cls.base_dir, ignore_errors=True)
        os.makedirs(cls.tmp_dir)

        cls._create_and_write_data()
        cls._train_model()
        cls._export_model()
        cls._predict_c_rf()

        pass

    @classmethod
    def tearDown(cls):
        # Remove temporary files
        shutil.rmtree(cls.base_dir, ignore_errors=True)
        pass

    @classmethod
    def _create_and_write_data(cls):
        np.random.seed(1)
        class_A_means = np.arange(10)
        class_B_means = np.arange(1, 11)
        feature_names = [f'F{i}' for i in range(10)]
        cls.feature_names = feature_names

        class_A_train = np.random.normal(class_A_means, size=(1000, 10))
        class_B_train = np.random.normal(class_B_means, size=(1000, 10))
        class_A_test = np.random.normal(class_A_means - 1, size=(1000, 10))
        class_B_test = np.random.normal(class_B_means + 1, size=(1000, 10))

        fold_foo = np.ones(2000, dtype=int)
        labels = np.array((["A"] * 1000) + (["B"] * 1000))

        columns = [cls.fold_column] + feature_names + [cls.label_col]

        df_train = pd.DataFrame(index=range(2000), columns=columns)
        df_train[cls.fold_column] = fold_foo
        df_train[feature_names] = np.vstack((class_A_train, class_B_train))
        df_train[cls.label_col] = labels

        df_test = pd.DataFrame(index=range(2000), columns=columns)
        df_test[cls.fold_column] = fold_foo
        df_test[feature_names] = np.vstack((class_A_test, class_B_test))
        df_test[cls.label_col] = labels

        df_train.to_csv(cls.train_file, index=False)
        df_test.to_csv(cls.test_file, index=False)

        cls.df_train = df_train
        cls.df_test = df_test

    @classmethod
    def _train_model(cls):
        cls._model = RandomForestClassifier(n_estimators=30, max_depth=5)
        train_x = cls.df_train[cls.feature_names]
        cls._model.fit(train_x, cls.df_train["LABEL"])
        cls._wrapper = AutonSKRFWrapper(cls._model)
        cls._wrapper.set_bounds_data(train_x)

    @classmethod
    def _export_model(cls):
        cls._wrapper.write_legacy(cls.model_file, "model_name")

    @classmethod
    def _predict_c_rf(cls):
        predict_command = ' '.join([
            cls._c_ernie,
            'option predict',
            f'load {cls.model_file}',
            f'testds {cls.test_file}',
            f'output_prefix {cls.pred_prefix}',
        ])
        ret = os.system(predict_command)
        if ret > 0:
            raise RuntimeError(f"Non-zero return when calling C AutonRF predict command: {predict_command}")

        cls._test_metrics = pd.read_csv(cls.pred_prefix + cls.metrics_suffix)
        cls._test_preds = pd.read_csv(cls.pred_prefix + cls.preds_suffix)
        cls._test_attr = pd.read_csv(cls.pred_prefix + cls.attr_suffix)

    def test_mean_entropy(self):
        c_mean_ent = self._test_metrics["Mean_Entropy"]
        sk_mean_ent = self._wrapper.mean_entropy(self.df_test[self.feature_names])
        self.assertTrue(np.allclose(c_mean_ent, sk_mean_ent, atol=.0001), "Mean entropy does not match")

    def test_dot_product_sum(self):
        c_dot_prod_sum = self._test_metrics["Dotproduct_Sum"]
        sk_dot_product_sum = self._wrapper.dot_product_sum(self.df_test[self.feature_names])
        self.assertTrue(np.allclose(c_dot_prod_sum, sk_dot_product_sum, atol=.001), "Dot product sum does not match")

    def test_inbounds(self):
        c_inbounds = self._test_metrics["Inbounds_score"]
        sk_inbounds = self._wrapper.inbounds(self.df_test[self.feature_names])
        self.assertTrue(np.allclose(c_inbounds, sk_inbounds, atol=.001), "Inbounds does not match")

    def test_feature_usage(self):
        c_dp = self._test_attr["DPCount"][1:-1]
        c_weighted = self._test_attr["WeightedCount"][1:-1]
        c_raw = self._test_attr["RawCount"][1:-1]
        sk_dp, sk_weighted, sk_raw = self._wrapper.feature_usage(self.df_test[self.feature_names])

        self.assertTrue(np.array_equal(c_dp, sk_dp))
        self.assertTrue(np.allclose(c_weighted, sk_weighted, atol=.0001))
        self.assertTrue(np.array_equal(c_raw, sk_raw))

    def test_predictions_match(self):
        preds_c = np.exp(self._test_preds[["A","B"]])/30
        preds_sk = self._model.predict_proba(self.df_test[self.feature_names])

        self.assertTrue(np.allclose(preds_c, preds_sk, atol=.0001))





