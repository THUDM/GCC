from gcc.models.emb import (
    FromNumpy,
    FromNumpyAlign,
    FromNumpyGraph,
    GraphWave,
    ProNE,
    Zero,
)


def build_model(name, hidden_size, **model_args):
    """
    Builds a model : classifier.

    Args:
        name: (str): write your description
        hidden_size: (int): write your description
        model_args: (dict): write your description
    """
    return {
        "zero": Zero,
        "from_numpy": FromNumpy,
        "from_numpy_align": FromNumpyAlign,
        "from_numpy_graph": FromNumpyGraph,
        "prone": ProNE,
        "graphwave": GraphWave,
    }[name](hidden_size, **model_args)
