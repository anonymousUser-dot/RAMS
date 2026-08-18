import importlib

from torch.utils.data import Dataset, DataLoader
from utils.ExpConfigs import ExpConfigs

def data_provider(configs: ExpConfigs, flag: str, shuffle_flag: bool = None, drop_last: bool = None) -> tuple[Dataset, DataLoader]:
    '''
    - flag: "train", "val", "test", "test_all"
    - shuffle_flag: In rare cases, it can be manually overwrite.
    - drop_last: In rare cases, it can be manually overwrite.
    '''
    # backward compatibility
    assert not (shuffle_flag or drop_last), "Please use --train_val_loader_shuffle 0/1 and --train_val_loader_drop_last 0/1 to set shuffle_flag and drop_last for train/val dataloader instead."
    # dynamically import the desired dataset class
    dataset_module = importlib.import_module(f"data.data_provider.datasets.{configs.dataset_name}")
    Data = dataset_module.Data

    # try to load custom collate_fn for the dataset, if present
    try:
        collate_fn = getattr(dataset_module, configs.collate_fn)
    except:
        collate_fn = None

    if flag in ["test", "test_all"]:
        shuffle_flag = False
        drop_last = False
        batch_size = configs.batch_size
    else:
        shuffle_flag = True if configs.train_val_loader_shuffle is None else bool(configs.train_val_loader_shuffle)
        drop_last = True if configs.train_val_loader_drop_last is None else bool(configs.train_val_loader_drop_last)
        batch_size = configs.batch_size

    data_set: Dataset = Data(
        configs=configs,
        flag=flag,
        # DEBUG: temporal change
        # **configs._asdict()
    )
    num_workers = max(0, int(configs.num_workers))
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle_flag,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
        "pin_memory": bool(configs.use_gpu and configs.pin_memory),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(configs.persistent_workers)
        loader_kwargs["prefetch_factor"] = max(1, int(configs.prefetch_factor))

    data_loader = DataLoader(data_set, **loader_kwargs)
    return data_set, data_loader
