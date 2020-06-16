from gcc.models.emb import (
    Zero,
    FromNumpy,
    FromNumpyAlign,
    FromNumpyGraph,
    ProNE,
    GraphWave,
)


def build_model(name, hidden_size, **model_args):
    return {
        "zero": Zero,
        "from_numpy": FromNumpy,
        "from_numpy_align": FromNumpyAlign,
        "from_numpy_graph": FromNumpyGraph,
        "prone": ProNE,
        "graphwave": GraphWave,
    }[name](hidden_size, **model_args)
