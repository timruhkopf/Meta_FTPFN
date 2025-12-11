import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def hidden_dim():
    """Hidden dimension for testing."""
    return 64


@pytest.fixture
def batch_size():
    """Batch size for testing."""
    return 2


@pytest.fixture
def seq_len():
    """Sequence length for testing."""
    return 10


@pytest.fixture
def num_contexts():
    """Number of context sentences."""
    return 3


@pytest.fixture
def num_heads():
    """Number of attention heads."""
    return 4


@pytest.fixture
def dummy_decoder(hidden_dim, num_heads):
    """Create a dummy decoder for testing."""
    class DummyDecoder(nn.Module):
        def __init__(self, hidden_dim, num_layers=4):
            super().__init__()
            self.embedding = nn.Embedding(100, hidden_dim)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    batch_first=True,
                    dim_feedforward=hidden_dim * 4,
                )
                for _ in range(num_layers)
            ])
            self.output = nn.Linear(hidden_dim, 100)
        
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
            return self.output(x)
    
    return DummyDecoder(hidden_dim)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def dummy_ft_batch(T=32, B=8, D=5, Tsplit=25):
    """
    Create a dummy batch of input data.
    
    returns:
        x: tuple of (features, targets)
        Tsplit: int, split index between train and test
    """ 
    torch.manual_seed(42)

    assert D >= 1 and D <= 11  # 1 int + up to 10 float features

    # First feature: integer in [0, 1000]
    ints = torch.randint(low=0, high=1001, size=(T, B))  # [T, B, 1][web:1]
    floats = torch.rand(T, B, D)  # [T, B, D-1][web:5]

    x_train = floats[:Tsplit]
    x_test = floats[Tsplit:]
    y_train = ints[:Tsplit].float()


    # this is the target format: 
    x = (
            torch.cat([x_train, x_test], dim=0),
            y_train
    )
    # single_eval_pos=x_train.shape[0],
    # src_key_padding_mask=None

    return x, Tsplit


@pytest.fixture
def ft_pfn():
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from ifbo.surrogate import FTPFN

    load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")

    model_path = os.getenv("MODELDIR") + "pfn_ckpt"
    frozen_model = FTPFN(
        target_path=Path(model_path), version="0.0.1", device="cpu"
    ).model

    criterion = frozen_model.criterion
    
    return frozen_model, criterion
