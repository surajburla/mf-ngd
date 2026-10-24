import numpy as np
import torch
from sklearn.metrics import average_precision_score
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import SpecCNN, SpecDAE, count_trainable
# ... rest of your imports and script code
from src.models import SpecCNN, SpecDAE, count_trainable
from src.utils import cluster_bootstrap, recall_at_threshold, select_threshold_at_precision


def test_param_count():
    assert count_trainable(SpecCNN("group")) == 43_713
    assert count_trainable(SpecCNN("batch")) == 43_713
    assert count_trainable(SpecCNN("none")) == 43_521


def test_shapes():
    x = torch.randn(4, 1, 216)
    assert SpecDAE()(x).shape == x.shape
    assert SpecCNN()(x).shape == (4, 1)


def test_groupnorm_per_sample_independence():
    torch.manual_seed(0)
    m = SpecCNN("group").eval()
    x = torch.randn(8, 1, 216)
    full = m(x)
    single = torch.cat([m(x[i:i + 1]) for i in range(8)])
    assert torch.allclose(full, single, atol=1e-5)


def test_batchnorm_breaks_independence():
    torch.manual_seed(0)
    m = SpecCNN("batch", dropout=0.0).train()
    x = torch.randn(8, 1, 216)
    assert not torch.allclose(m(x)[:1], m(x[:1]), atol=1e-5)


def test_threshold_and_bootstrap():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(500), np.ones(500)].astype(int)
    s = np.r_[rng.uniform(0, .6, 500), rng.uniform(.4, 1, 500)]
    thr = select_threshold_at_precision(y, s, 0.9)
    assert not np.isnan(thr) and recall_at_threshold(y, s, thr) > 0.3
    groups = np.tile(np.arange(10), 100)
    ci = cluster_bootstrap(average_precision_score, y, s, groups, n_boot=200)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    d = cluster_bootstrap(average_precision_score, y, s, groups, scores_b=s, n_boot=200)
    assert abs(d["point"]) < 1e-12 and abs(d["lo"]) < 1e-12 and abs(d["hi"]) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
