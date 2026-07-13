import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import src.data_loader as dl


class BigQueryTimeoutTests(unittest.TestCase):
    def test_query_uses_timeout_and_cancels_on_timeout(self):
        job = Mock()
        job.result.side_effect = TimeoutError("did not finish")

        client = Mock()
        client.query.return_value = job
        client.get_table.return_value = Mock(schema=[])

        with patch.dict(os.environ, {"BQ_QUERY_TIMEOUT_SECONDS": "7"}, clear=False):
            with patch.object(dl, "get_bigquery_client", return_value=client):
                with patch.object(dl, "_download_df", return_value="done"):
                    with self.assertRaisesRegex(RuntimeError, "timed out"):
                        dl._query_bigquery(limit=5)

        job.result.assert_called_once_with(timeout=7)
        job.cancel.assert_called_once()

    def test_property_performance_query_uses_timeout_and_cancels_on_timeout(self):
        job = Mock()
        job.result.side_effect = TimeoutError("did not finish")

        client = Mock()
        client.query.return_value = job
        client.get_table.return_value = Mock(
            schema=[
                SimpleNamespace(name="businessDay"),
                SimpleNamespace(name="houseCount"),
            ]
        )

        with patch.dict(os.environ, {"BQ_QUERY_TIMEOUT_SECONDS": "9"}, clear=False):
            with patch.object(dl, "get_bigquery_client", return_value=client):
                with patch.object(dl, "_download_df", return_value="done"):
                    with self.assertRaisesRegex(RuntimeError, "timed out"):
                        dl._query_property_performance(limit=5)

        job.result.assert_called_once_with(timeout=9)
        job.cancel.assert_called_once()


if __name__ == "__main__":
    unittest.main()
