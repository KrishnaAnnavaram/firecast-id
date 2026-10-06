"""firecast-id: daily fire hotspot forecasts for Indonesia with leak-free backtests."""
import warnings as _warnings

# joblib warns on some Windows hosts when it cannot count physical cores. The count is not used here.
_warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

__version__ = "0.1.0"
