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
