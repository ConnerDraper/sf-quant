"""Shared helpers for the SLURM worker."""

from sf_quant.optimizer import Constraint, LongOnly, ZeroBeta, ZeroInvestment

#: Constraints addressable by name from a YAML config. All are stated in active
#: space; see ``sf_quant.optimizer.constraints`` for why total-space names like
#: FullInvestment and UnitBeta no longer exist.
CONSTRAINT_REGISTRY: dict[str, type] = {
    "ZeroInvestment": ZeroInvestment,
    "ZeroBeta": ZeroBeta,
    "LongOnly": LongOnly,
}


def get_constraints(constraint_names: list[str]) -> list[Constraint]:
    """Convert constraint names to constraint objects.

    Args:
        constraint_names: List of constraint class names.

    Returns:
        List of instantiated constraint objects.

    Raises:
        KeyError: If a constraint name is not in the registry.
    """
    constraints = []
    for name in constraint_names:
        if name not in CONSTRAINT_REGISTRY:
            available = ", ".join(CONSTRAINT_REGISTRY)
            raise KeyError(f"Unknown constraint '{name}'. Available: {available}")
        constraints.append(CONSTRAINT_REGISTRY[name]())
    return constraints
