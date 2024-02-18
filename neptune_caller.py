from typing import Any
from pathlib import Path
from getpass import getpass

from dotenv import dotenv_values
import neptune

from .plotter import History

def get_neptune_key() -> str:
    file = Path(".env")
    if not file.exists() or "NEPTUNE_API_KEY" not in dotenv_values(file):
        key = input("No .env was found, please input the key (it won't be displayed):\n")
        if not key:
            raise RuntimeError("No key was provided, aborting")
        
        with file.open("w") as f:
            print(f'NEPTUNE_API_KEY="{key}"', file=f)
            
    return dotenv_values(file)["NEPTUNE_API_KEY"]
    
def get_neptune_run(name: str, description: str, init_kwargs: dict[str, Any],
                    *, project_name: str) -> neptune.Run:
    """
    Get Neptune session with specified name and description
    
    :param name: name of experiment
    :param description: nep
    :param init_kwargs: additional parameters passed to `neptune.init_run`, such as `tags=<list>`, `with_id=<str>`
    :param project_name:
    """
    api_key = get_neptune_key()
    
    neptune_run = neptune.init_run(
        project=project_name,
        api_token=api_key,
        name=name,
        description=description,
        **init_kwargs
    )
    return neptune_run

def push_metrics(history: History, neptune_run: neptune.Run):
    """
    Push last metrics in history into neptune experiment
    """
    for handle, metrics in ("tr", history.train[-1]), ("val", history.val[-1]):
        for metric_name, metric_val in metrics.items():
            neptune_run[f"metrics/{handle}/{metric_name}"].append(metric_val)
