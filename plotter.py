import warnings
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, Image
from plotly.subplots import make_subplots
from tqdm.auto import tqdm


Report = dict[str, float]
@dataclass
class History:
    train: list[Report] = field(default_factory=list)
    val: list[Report] = field(default_factory=list)
    drop_query: str = 'phase != phase' # query to return none
    
    def push_epoch(self, *reports: tuple[Report, Report] | tuple[Report]) -> None:
        if len(reports) == 1:
            val_report, = reports
            train_report = {}
        else:
            train_report, val_report = reports
        self.train.append(train_report)
        self.val.append(val_report)
    
    def as_dfs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # return metrics as 2 DataFrames with additional epoch and phase (train / val) columns
        # for plotting
        
        df = pd.concat(map(pd.DataFrame, [self.train, self.val]),
                       keys=["train", "val"])\
                .reset_index(names=["phase", "epoch"])
        df.epoch += 1
        
        return df.query(f"not ({self.drop_query})"), df.query(self.drop_query)
    
    def as_df(self) -> pd.DataFrame:
        return self.as_dfs()[0]
    
    def __len__(self) -> int:
        return len(self.train)
    
    def range(self, max_epochs, stopper = None, do_tqdm = False):
        loop = range(max(max_epochs - len(self), 0))
        if do_tqdm:
            loop = tqdm(loop)
            
        for i in loop:
            if stopper is not None and stopper:
                return
            yield i
    

@dataclass
class Plotter:
    metrics: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    height: int = 600
    width: int = 1000
    plot_cols: int = 3
    path: Path | str | None = None
    bound_history: History | None = None
    # custom_range: dict[str, tuple[int, int]] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.path is not None:
            self.path = Path(self.path)
            if self.path.exists():
                warnings.warn("Warning...........Message")
            
        self._inited = False
        if not self.titles:
            self.titles = self.metrics
        
        self.fig: plotly.graph_objs._figure.Figure
        self.traces = {}
        self.names = set()
        
    def init_canvas(self):
        plot_cols = min(self.plot_cols, len(self.metrics))
        plot_rows = max(1, ceil(len(self.metrics) / plot_cols))
        self.fig = go.FigureWidget(make_subplots(rows=plot_rows, cols=plot_cols, subplot_titles=self.titles))
        self.fig.update_layout(height=self.height, width=self.width, showlegend=True)
        display(self.fig)
    
    def display(self):
        display(self.fig)
    
    def plot(self, history: History | None = None):
        if history is None:
            history = self.bound_history
            assert history is not None
        
        df_kept, df_dropped = history.as_dfs()
        if not self._inited:
            if not self.metrics:
                self.metrics = self.titles = list(set(df_dropped.columns) - {"phase", "epoch"})
            self.init_canvas()
            self._inited = True
        
        for handle, df in ("kept", df_kept), ("dropped", df_dropped):
            for i, metric in enumerate(self.metrics):
                for trace in px.scatter(df.dropna(subset=metric),
                                        x="epoch", y=metric, color="phase")["data"]:
                    
                    if (handle, metric, trace.name) in self.traces:
                        _trace = self.traces[(handle, metric, trace.name)]
                        # avoid unneeded assignments, for they cause lags
                        if _trace.y.shape != trace.y.shape:
                            _trace.x, _trace.y = trace.x, trace.y
                        trace = _trace
                    else:
                        self.fig.append_trace(
                            trace,
                            row=i // self.plot_cols + 1, col=i % self.plot_cols + 1
                        )
                        self.traces[(handle, metric, trace.name)] = trace = self.fig.data[-1]
                        # remove duplicate legends
                        if trace.name in self.names:
                            trace.update(showlegend=False)
                        else:
                            self.names.add(trace.name)
                            
                    if trace.x.size > 1 and trace.mode != "lines" and handle != "dropped":
                        trace.mode = "lines"
                        
                    if handle == "dropped" and ("kept", metric, trace.name) in self.traces:
                        kept = self.traces[("kept", metric, trace.name)]
                        y = np.full_like(trace.y, kept.y[0])
                        
                        if not all(trace.y == y):
                            trace.customdata = trace.y[:, None]
                            trace.y = y
                            trace.hovertemplate = trace.hovertemplate.replace(
                                "%{y}",
                                "%{customdata[0]:.2f}" + f"({history.drop_query})"
                            )
                            trace.mode = "markers"
                            trace.marker["symbol"] = "star"
                            trace.marker["size"] = 10
                            trace.marker["color"] = kept.marker["color"]
                    
        if self.path is not None:
            self.fig.write_image(self.path)
            self.fig.write_html(str(self.path) + ".html")
            
    def draw_no_widget(self):
        return plotly.io.from_json(self.fig.to_json())

def plotly_static(fig: plotly.graph_objs._figure.Figure, format: str = "png") -> Image:
    return Image(fig.to_image(format=format))
