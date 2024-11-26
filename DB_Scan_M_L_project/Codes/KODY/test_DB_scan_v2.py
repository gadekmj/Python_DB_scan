import unittest
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score
from DB_SCAN_V2 import CustomDBSCAN, find_best_dbscan_params

class TestCustomDBSCANV2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the digits dataset
        cls.digits = datasets.load_digits()
        cls.n_samples = len(cls.digits.images)
        cls.data = cls.digits.images.reshape((cls.n_samples, -1))

        # Apply PCA to reduce dimensionality
        cls.pca = PCA(n_components=12)
        cls.data_pca = cls.pca.fit_transform(cls.data)

        # Split data into 50% train and 50% test subsets
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.data_pca, cls.digits.target, test_size=0.5, shuffle=False
        )

    def test_data_preprocessing(self):
        """Test if data is correctly loaded and preprocessed"""
        self.assertEqual(self.data.shape, (self.n_samples, 64))
        self.assertEqual(self.data_pca.shape[1], 12)

    def test_fit(self):
        """Test the fit method"""
        dbscan = CustomDBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(self.X_test)
        self.assertIsNotNone(dbscan.labels)
        self.assertEqual(len(dbscan.labels), len(self.X_test))

    def test_get_neighbors(self):
        """Test the _get_neighbors method"""
        dbscan = CustomDBSCAN(eps=0.5, min_samples=5)
        neighbors = dbscan._get_neighbors(self.X_test, 0)
        self.assertGreaterEqual(len(neighbors), 0)

    def test_expand_cluster(self):
        """Test the _expand_cluster method"""
        dbscan = CustomDBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(self.X_test)  # Ensure labels are initialized
        neighbors = dbscan._get_neighbors(self.X_test, 0)
        dbscan._expand_cluster(self.X_test, 0, neighbors, 0)
        self.assertIn(0, dbscan.labels)

    def test_fit_predict(self):
        """Test the fit_predict method"""
        dbscan = CustomDBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(self.X_test)
        self.assertIsNotNone(labels)
        self.assertEqual(len(labels), len(self.X_test))

    def test_find_best_dbscan_params(self):
        """Test parameter optimization process"""
        result = find_best_dbscan_params(self.X_test, self.y_test, 0.1, 0.5, 0.1, range(2, 5), use_custom=True)
        self.assertIn("best_eps", result)
        self.assertIn("best_min_samples", result)
        self.assertIn("best_score", result)

if __name__ == '__main__':
    unittest.main(verbosity=2)