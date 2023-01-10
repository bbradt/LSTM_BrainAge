from dataloaders.fake_fnc import FakeFNC
from dataloaders.load_dev_data import UKBData
from dataloaders.load_UKB_HCP1200 import UKBHCP1200Data
from dataloaders.cadasil import CadasilData


def get_dataset(key, *args, **kwargs):
    if key.lower() == "fakefnc":
        return FakeFNC(*args, **kwargs)
    elif key.lower() == "ukb":
        return UKBData(*args, **kwargs)
    elif key.lower() == "ukbhcp1200":
        return UKBHCP1200Data(*args, **kwargs)
    elif key.lower() == "cadasil":
        return CadasilData(*args, **kwargs)
