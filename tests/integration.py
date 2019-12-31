import unittest
from saliency.client import SaliencyClient


class TestClient(unittest.TestCase):
    def test_predict(self):
        model_name = "densenet_mura_rs_v3_xr_shoulder.h5"
        client = SaliencyClient("user", "superuser", "http://127.0.0.1:8000/")
        with open('setup.py', 'r') as f:
            pfile = ("setup.py", f, 'application/x-python-code')
            datapoint = client.create_datapoint(
                {"filename": "test.png", "meta": "{}"}, pfile)
        client.predict(datapoint["id"], model_name)
        annotation = client.pop_annotation(
            {"datapoint_id": datapoint["id"], "model": model_name})
        datapoint = client.get_datapoint(datapoint["id"])
        annotation["status"] = "done"
        client.update_annotation(annotation)
        annotation = client.predict(datapoint["id"], model_name)
        client.delete_datapoint(datapoint["id"])


if __name__ == '__main__':
    unittest.main()
