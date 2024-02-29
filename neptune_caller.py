from typing import Any
import neptune

from .plotter import History
from .utils import get_dotenv_key
    
def get_neptune_run(name: str, description: str, init_kwargs: dict[str, Any],
                    *, project_name: str) -> neptune.Run:
    """
    Get Neptune session with specified name and description
    
    :param name: name of experiment
    :param description: nep
    :param init_kwargs: additional parameters passed to `neptune.init_run`, such as `tags=<list>`, `with_id=<str>`
    :param project_name:
    """
    api_key = get_dotenv_key("NEPTUNE_API_KEY")
    
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
