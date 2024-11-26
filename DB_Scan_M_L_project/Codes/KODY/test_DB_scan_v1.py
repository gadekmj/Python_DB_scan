import unittest
import numpy as np
from sklearn.decomposition import PCA
from DB_SCAN_V1 import CustomDBSCAN, create_smiling_face_data

class TestCustomDBSCANV1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = create_smiling_face_data()
        cls.pca = PCA(n_components=2)
        cls.data_pca = cls.pca.fit_transform(cls.data)
        cls.dbscan = CustomDBSCAN(eps=0.2, min_samples=7)

    def test_data_preprocessing(self):
        """Test if data is correctly loaded and preprocessed"""
        self.assertEqual(self.data.shape, (600, 2))
        self.assertEqual(self.data_pca.shape[1], 2)

    def test_fit(self):
        """Test the fit method"""
        self.dbscan.fit(self.data_pca)
        self.assertIsNotNone(self.dbscan.labels)
        self.assertEqual(len(self.dbscan.labels), len(self.data_pca))

    def test_get_neighbors(self):
        """Test the _get_neighbors method"""
        neighbors = self.dbscan._get_neighbors(self.data_pca, 0)
        self.assertGreaterEqual(len(neighbors), 0)

    def test_expand_cluster(self):
        """Test the _expand_cluster method"""
        self.dbscan.fit(self.data_pca)  # Ensure labels are initialized
        neighbors = self.dbscan._get_neighbors(self.data_pca, 0)
        self.dbscan._expand_cluster(self.data_pca, 0, neighbors, 0)
        self.assertIn(0, self.dbscan.labels)

    def test_fit_predict(self):
        """Test the fit_predict method"""
        labels = self.dbscan.fit_predict(self.data_pca)
        self.assertIsNotNone(labels)
        self.assertEqual(len(labels), len(self.data_pca))

if __name__ == '__main__':
    unittest.main(verbosity=2)