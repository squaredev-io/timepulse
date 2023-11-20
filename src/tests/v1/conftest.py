from pytest import fixture
from src.tests.utils.maps import v1_test_order_map
import logging, os, shutil

logging.getLogger("tensorflow").disabled = True
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("geopy").setLevel(logging.INFO)


def get_order_number(task):
    return v1_test_order_map.index(task)


"""
We are creating some fixtures here that are session scoped.
https://docs.pytest.org/en/6.2.x/example/special.html?highlight=session
And we are creating a test client for each api or dependency we need.
Fixtures can be accessed from any test by accepting the name of the 
fixture as a parameter in the test function
"""


# @fixture(scope="session", autouse=True)
# async def db():
#     # runs once before all tests
#     db = "this is a database"
#     yield db


@fixture(scope="session", autouse=True)
def setup_storage():
    yield True
    shutil.rmtree("storage")


class DataHolder:
    client = None
    client_id = None
    access_token = None
    user_id = None
    item_id = None

    batch_item_1 = "1"
    batch_item_2 = "2"


dataholder = DataHolder()
