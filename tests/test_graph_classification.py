import argparse
import os

from gcc.tasks.graph_classification import GraphClassification
from tests.utils import E2E_PATH, MOCO_PATH, get_default_args, generate_emb


def run(dataset, model, emb_path=""):
    args = get_default_args()
    args.dataset = dataset
    args.model = model
    args.emb_path = emb_path
    task = GraphClassification(
        args.dataset,
        args.model,
        args.hidden_size,
        args.num_shuffle,
        args.seed,
        emb_path=args.emb_path,
    )
    return task.train()


def test_e2e_imdb_binary():
    NAME = "imdb-binary"
    generate_emb(os.path.join(E2E_PATH, "current.pth"), NAME)
    ret = run(NAME, "from_numpy_graph", os.path.join(E2E_PATH, f"{NAME}.npy"))
    assert ret["Micro-F1"] > 0.7, ret


def test_e2e_imdb_multi():
    NAME = "imdb-multi"
    generate_emb(os.path.join(E2E_PATH, "current.pth"), NAME)
    ret = run(NAME, "from_numpy_graph", os.path.join(E2E_PATH, f"{NAME}.npy"))
    assert ret["Micro-F1"] > 0.48, ret


def test_e2e_collab():
    NAME = "collab"
    generate_emb(os.path.join(E2E_PATH, "current.pth"), NAME)
    ret = run(NAME, "from_numpy_graph", os.path.join(E2E_PATH, f"{NAME}.npy"))
    assert ret["Micro-F1"] > 0.73, ret


def test_e2e_rdt_b():
    NAME = "rdt-b"
    generate_emb(os.path.join(E2E_PATH, "current.pth"), NAME)
    ret = run(NAME, "from_numpy_graph", os.path.join(E2E_PATH, f"{NAME}.npy"))
    assert ret["Micro-F1"] > 0.85, ret


def test_e2e_rdt_5k():
    NAME = "rdt-5k"
    generate_emb(os.path.join(E2E_PATH, "current.pth"), NAME)
    ret = run(NAME, "from_numpy_graph", os.path.join(E2E_PATH, f"{NAME}.npy"))
    assert ret["Micro-F1"] > 0.50, ret


def test_moco_imdb_binary():
    NAME = "imdb-binary"
    generate_emb(os.path.join(MOCO_PATH, "current.pth"), NAME)
    ret = run(NAME, "from_numpy_graph", os.path.join(MOCO_PATH, f"{NAME}.npy"))
    assert ret["Micro-F1"] > 0.7, ret


def test_moco_imdb_multi():
    NAME = "imdb-multi"
    generate_emb(os.path.join(MOCO_PATH, "current.pth"), NAME)
    ret = run(NAME, "from_numpy_graph", os.path.join(MOCO_PATH, f"{NAME}.npy"))
    assert ret["Micro-F1"] > 0.48, ret


def test_moco_collab():
    NAME = "collab"
    generate_emb(os.path.join(MOCO_PATH, "current.pth"), NAME)
    ret = run(NAME, "from_numpy_graph", os.path.join(MOCO_PATH, f"{NAME}.npy"))
    assert ret["Micro-F1"] > 0.77, ret


def test_moco_rdt_b():
    NAME = "rdt-b"
    generate_emb(os.path.join(MOCO_PATH, "current.pth"), NAME)
    ret = run(NAME, "from_numpy_graph", os.path.join(MOCO_PATH, f"{NAME}.npy"))
    assert ret["Micro-F1"] > 0.88, ret


def test_moco_rdt_5k():
    NAME = "rdt-5k"
    generate_emb(os.path.join(MOCO_PATH, "current.pth"), NAME)
    ret = run(NAME, "from_numpy_graph", os.path.join(MOCO_PATH, f"{NAME}.npy"))
    assert ret["Micro-F1"] > 0.52, ret
