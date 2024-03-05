from pathlib import Path
from typing import Callable

import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset

class InRamDataset(Dataset):
    """
    Cache dataset in memory (because I have a ton of ram but somewhat slow HDD)
        for faster training.
    """
    def __init__(self, save_path: Path, dataset: Dataset | None):
        save_path = Path(save_path)
        self.save_path = save_path

        if not save_path.exists():
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True)
            
            entries = list(tqdm(dataset))
            transposed = [torch.nested.nested_tensor(list(column))
                          if isinstance(column[0], torch.Tensor)
                          else column
                          for column in zip(*entries)]
            
            torch.save(transposed, save_path)
        else:
            transposed = torch.load(save_path)
        self.data = transposed
        
        self.transform: Callable[[torch.Tensor], torch.Tensor] | None = None
    
    def __getitem__(self, index: int) -> tuple:
        entry = tuple(column[index] for column in self.data)
        if self.transform is not None:
            entry = entry
        return entry

    def __len__(self):
        column = self.data[0]
        return len(column) if isinstance(column, list) else column.size(0)

class InRamSubset(InRamDataset):
    """
    Used for splitting InRamDataset(e. g. for train and val sets)
    
    Doesn't copy data of initial dataset, only copies links / views (in case of nested_tensor)
        so memory overhead is insignificant.
    """
    def __init__(self, dataset: InRamDataset, indices: list[int] | None = None):
        super(Dataset, self).__init__()
        
        if indices is None:
            self.data = dataset.data
        else:
            self.data = [[column[idx]
                          for idx in indices]
                         for column in dataset.data]
        
        self.transform = dataset.transform
    
