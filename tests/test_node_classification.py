import argparse
import os

from gcc.tasks.node_classification import NodeClassification
from tests.utils import E2E_PATH, MOCO_PATH, get_default_args, generate_emb


def run(dataset, model, emb_path=""):
    args = get_default_args()
    args.dataset = dataset
    args.model = model
    args.emb_path = emb_path
    task = NodeClassification(
        args.dataset,
        args.model,
        args.hidden_size,
        args.num_shuffle,
        args.seed,
        emb_path=args.emb_path,
    )
    return task.train()


def test_prone_airport():
    ret = run("usa_airport", "prone")
    assert ret["Micro-F1"] > 0.62, ret["Micro-F1"]


def test_prone_hindex():
    ret = run("h-index", "prone")
    assert ret["Micro-F1"] > 0.69, ret["Micro-F1"]


def test_graphwave_airport():
    ret = run("usa_airport", "graphwave")
    assert ret["Micro-F1"] > 0.59, ret["Micro-F1"]


# def test_graphwave_hindex():
#     ret = run("h-index", "graphwave")
#     assert ret["Micro-F1"] > 0.70, ret["Micro-F1"]


def test_e2e_airport():
    generate_emb(os.path.join(E2E_PATH, "current.pth"), "usa_airport")
    ret = run("usa_airport", "from_numpy", os.path.join(E2E_PATH, "usa_airport.npy"))
    assert ret["Micro-F1"] > 0.62, ret["Micro-F1"]


def test_e2e_hindex():
    generate_emb(os.path.join(E2E_PATH, "current.pth"), "h-index")
    ret = run("h-index", "from_numpy", os.path.join(E2E_PATH, "h-index.npy"))
    assert ret["Micro-F1"] > 0.76, ret["Micro-F1"]


def test_moco_airport():
    generate_emb(os.path.join(MOCO_PATH, "current.pth"), "usa_airport")
    ret = run("usa_airport", "from_numpy", os.path.join(MOCO_PATH, "usa_airport.npy"))
    assert ret["Micro-F1"] > 0.63, ret["Micro-F1"]


def test_moco_hindex():
    generate_emb(os.path.join(MOCO_PATH, "current.pth"), "h-index")
    ret = run("h-index", "from_numpy", os.path.join(MOCO_PATH, "h-index.npy"))
    assert ret["Micro-F1"] > 0.73, ret["Micro-F1"]
