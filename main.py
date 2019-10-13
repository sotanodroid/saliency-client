import sys
from saliency.saliency import SaliencyClient

_, username, password = sys.argv

host = 'localhost:8000/'

login = dict(
    username=username,
    password=password,
    host=host
)

sapi = SaliencyClient(**login)
