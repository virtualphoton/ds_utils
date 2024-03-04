from pathlib import Path
from typing import Callable

import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset

class InRamDataset(Dataset):
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
        column = self.data[0]
        self.length = len(column) if isinstance(column, list) else column.size(0)
        
        self.transform: Callable[[torch.Tensor], torch.Tensor] | None = None
    
    def __getitem__(self, index: int):
        entry = tuple(column[index] for column in self.data)
        if self.transform is not None:
            entry = entry
        return entry

    def __len__(self):
        return self.length
