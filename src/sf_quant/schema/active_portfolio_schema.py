import dataframely as dy


class ActivePortfolioSchema(dy.Schema):
    date = dy.Date(nullable=False)
    barrid = dy.String(nullable=False)
    active_weight = dy.Float64(nullable=False)
    benchmark_weight = dy.Float64(nullable=False)
    active_risk = dy.Float64(nullable=False)
