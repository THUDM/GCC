import argparse
import os

from gcc.tasks.node_classification import NodeClassification
from tests.utils import E2E_PATH, MOCO_PATH, get_default_args


def run_for_args(args):
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
    args = get_default_args()
    args.dataset = "usa_airport"
    args.model = "prone"
    args.emb_path = ""
    ret = run_for_args(args)

    assert ret["Micro-F1 "] > 0.62, ret["Micro-F1 "]


def test_prone_hindex():
    args = get_default_args()
    args.dataset = "h-index"
    args.model = "prone"
    args.emb_path = ""
    ret = run_for_args(args)

    assert ret["Micro-F1 "] > 0.69, ret["Micro-F1 "]


def test_graphwave_airport():
    args = get_default_args()
    args.dataset = "usa_airport"
    args.model = "graphwave"
    args.emb_path = ""
    ret = run_for_args(args)

    assert ret["Micro-F1 "] > 0.59, ret["Micro-F1 "]


def test_e2e_airport():
    args = get_default_args()
    args.dataset = "usa_airport"
    args.model = "from_numpy"
    args.emb_path = os.path.join(E2E_PATH, "usa_airport.npy")
    ret = run_for_args(args)

    assert ret["Micro-F1 "] > 0.62, ret["Micro-F1 "]


def test_e2e_hindex():
    args = get_default_args()
    args.dataset = "h-index"
    args.model = "from_numpy"
    args.emb_path = os.path.join(E2E_PATH, "h-index.npy")
    ret = run_for_args(args)

    assert ret["Micro-F1 "] > 0.76, ret["Micro-F1 "]


def test_moco_airport():
    args = get_default_args()
    args.dataset = "usa_airport"
    args.model = "from_numpy"
    args.emb_path = os.path.join(MOCO_PATH, "usa_airport.npy")
    ret = run_for_args(args)

    assert ret["Micro-F1 "] > 0.63, ret["Micro-F1 "]


def test_moco_hindex():
    args = get_default_args()
    args.dataset = "h-index"
    args.model = "from_numpy"
    args.emb_path = os.path.join(MOCO_PATH, "h-index.npy")
    ret = run_for_args(args)

    assert ret["Micro-F1 "] > 0.73, ret["Micro-F1 "]


# def test_graphwave_hindex():
#     args = get_default_args()
#     args.dataset = "h-index"
#     args.model = "graphwave"
#     args.emb_path = ""
#     ret = run_for_args(args)

#     assert ret['Micro-F1 '] > 0.70, ret['Micro-F1 ']
