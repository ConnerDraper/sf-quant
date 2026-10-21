"""
The ``performance`` module provides tools for evaluating and summarizing
portfolio and strategy performance. It exposes a clean public API for
calculating returns, risk metrics, and visualizing performance, while keeping
internal computation details hidden.
"""

from .returns import (
    generate_active_returns_from_weights,
    generate_returns_from_weights,
    generate_multi_returns_from_weights,
    generate_cumulative_log_returns,
)
from .annualize import infer_periods_per_year
from .transfer import generate_transfer_coefficients
from .leverage import generate_leverage_from_weights
from .drawdown import generate_drawdown_from_returns
from .tables import (
    generate_returns_summary_table,
    generate_multi_returns_summary_table,
    generate_leverage_summary_table,
    generate_drawdown_summary_table,
)
from .charts import (
    generate_returns_chart,
    generate_multi_returns_chart,
    generate_leverage_chart,
    generate_drawdown_chart,
    generate_ic_chart,
)
from .ics import (
    generate_alpha_ics,
)
from .turnover import (
    get_turnover_stats,
    plot_turnover,
)

__all__ = [
    # Returns
    "generate_active_returns_from_weights",
    "generate_returns_from_weights",
    "generate_multi_returns_from_weights",
    "generate_cumulative_log_returns",
    "infer_periods_per_year",
    "generate_transfer_coefficients",
    # Leverage
    "generate_leverage_from_weights",
    # Drawdown
    "generate_drawdown_from_returns",
    # Summary tables
    "generate_returns_summary_table",
    "generate_multi_returns_summary_table",
    "generate_leverage_summary_table",
    "generate_drawdown_summary_table",
    # Charts
    "generate_returns_chart",
    "generate_multi_returns_chart",
    "generate_leverage_chart",
    "generate_drawdown_chart",
    # Alpha
    "generate_alpha_ics",
    "generate_ic_chart",
    # Turnover
    "get_turnover_stats",
    "plot_turnover",
]
