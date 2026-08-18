import torch
from pathlib import Path
from sklearn import model_selection

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs
from data.dependencies.WearableActivity import WearableActivity, Wearable_time_chunk
from data.data_provider.datasets.HumanActivity import (
    collate_fn,
    collate_fn_patch,
    collate_fn_tpatch,
)


class Data:
    def __init__(self, configs: ExpConfigs, flag: str = "train", **kwargs):
        logger.debug(f"getting {flag} set of MHEALTH")
        self.configs = configs
        assert flag in ["train", "test", "val", "test_all"]
        self.flag = flag
        self.preprocess()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def preprocess(self):
        cache_path = Path(self.configs.dataset_root_path) / "processed" / f"chunks_sl{self.configs.seq_len}_pl{self.configs.pred_len}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            train_data, val_data, test_data = cached["train"], cached["val"], cached["test"]
        else:
            dataset = WearableActivity(root=self.configs.dataset_root_path, dataset="MHEALTH")
            seen_data, test_data = model_selection.train_test_split(dataset, train_size=0.8, random_state=42, shuffle=False)
            train_data, val_data = model_selection.train_test_split(seen_data, train_size=0.875, random_state=42, shuffle=False)
            train_data = Wearable_time_chunk(train_data, self.configs)
            val_data = Wearable_time_chunk(val_data, self.configs)
            test_data = Wearable_time_chunk(test_data, self.configs)
            torch.save({"train": train_data, "val": val_data, "test": test_data}, cache_path)
        all_data = train_data + val_data + test_data
        self._set_max_lengths(all_data)
        if self.flag == "test_all":
            self.data = all_data
        elif self.flag == "train":
            self.data = train_data
        elif self.flag == "val":
            self.data = val_data
        else:
            self.data = test_data

    def _set_max_lengths(self, data):
        self.configs.seq_len_max_irr = max((sample["x"].shape[0] for sample in data), default=self.configs.seq_len)
        self.configs.pred_len_max_irr = max((sample["y"].shape[0] for sample in data), default=self.configs.pred_len)
        self.configs.patch_len_max_irr = max(self.configs.seq_len_max_irr, self.configs.pred_len_max_irr)
