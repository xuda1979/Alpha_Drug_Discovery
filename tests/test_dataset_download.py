import unittest
import importlib
from unittest.mock import patch, MagicMock
import io
import zipfile

try:
    pd = importlib.import_module('pandas')
except Exception as e:  # pragma: no cover - skip if deps missing
    pd = None
    IMPORT_ERROR = e

class TestDatasetDownload(unittest.TestCase):
    def test_download_chembl_activity_data(self):
        if pd is None:
            self.skipTest(f'Dependencies missing: {IMPORT_ERROR}')
        module = importlib.import_module('data.dataset_download')

        sample_json = {"activities": [{"activity_id": 1, "standard_value": "10"}]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_json
        mock_resp.raise_for_status.return_value = None
        with patch('data.dataset_download.requests.get', return_value=mock_resp) as m:
            df = module.download_chembl_activity_data('CHEMBL25', limit=1)
            self.assertEqual(len(df), 1)
            self.assertIn('activity_id', df.columns)

    def test_download_bindingdb_dataset(self):
        if pd is None:
            self.skipTest(f'Dependencies missing: {IMPORT_ERROR}')
        module = importlib.import_module('data.dataset_download')

        # create sample tsv and zip it in memory
        sample_tsv = 'colA\tcolB\n1\t2\n'
        mem_zip = io.BytesIO()
        with zipfile.ZipFile(mem_zip, 'w') as zf:
            zf.writestr('BindingDB_All.tsv', sample_tsv)
        mem_zip.seek(0)

        mock_resp = MagicMock()
        mock_resp.content = mem_zip.getvalue()
        mock_resp.raise_for_status.return_value = None

        with patch('data.dataset_download.requests.get', return_value=mock_resp):
            with patch('data.dataset_download.os.path.exists', return_value=False):
                df = module.download_bindingdb_dataset(path='BindingDB_All.tsv', use_sample=True)
                self.assertEqual(len(df), 1)
                self.assertIn('colA', df.columns)

if __name__ == '__main__':
    unittest.main()
