import numpy as np
import torch
from pathlib import Path
from typing import Tuple

BASEDIR = Path(__file__).parent

data = np.genfromtxt(
    f"{BASEDIR}/data/wdbc.data",
    delimiter=",", dtype="str")


def cast_to_int(data):
    return torch.from_numpy(data.astype("float32"))


def exclude_column(data, *column_number: Tuple[int]):
    data_copy = data.copy()
    data_copy = np.delete(data_copy, column_number, 1)
    return data_copy


target = np.where(data[:, 1] == "M", 1, 0)
col1 = exclude_column(data, 1)

target_tensor = torch.from_numpy(target).reshape(-1, 1)
print(target_tensor.shape)
data_tensor = cast_to_int(col1)
# print(data_tensor.shape)

# print(torch.cat([target_tensor, data_tensor], dim=1))
print("mean: ", data_tensor.mean(dim=0))
print("max: ", data_tensor.max(dim=0))
print("std: ", data_tensor.std(dim=0))
print("min: ", data_tensor.min(dim=0))
