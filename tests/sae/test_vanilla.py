import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

# Import the classes to be tested
from nanosae.sae.vanilla import VanillaSAE, VanillaSAETrainingWrapper, TrainStepOutput
from nanosae.utils.device import get_device, set_device

set_device("cpu")

@pytest.fixture
def sae_params():
    return {
        'd_in': 10,
        'd_sae': 5,
        'device': get_device()
    }

@pytest.fixture
def vanilla_sae(sae_params):
    return VanillaSAE(**sae_params)

def test_vanilla_sae_initialization(vanilla_sae, sae_params):
    assert vanilla_sae.d_in == sae_params['d_in']
    assert vanilla_sae.d_sae == sae_params['d_sae']
    assert vanilla_sae.W_enc.shape == (sae_params['d_in'], sae_params['d_sae'])
    assert vanilla_sae.b_enc.shape == (sae_params['d_sae'],)
    assert vanilla_sae.b_dec.shape == (sae_params['d_in'],)

def test_vanilla_sae_tied_weights(sae_params):
    sae_tied = VanillaSAE(**sae_params, tied_weights=True)
    assert sae_tied._W_dec is None
    assert_close(sae_tied.W_dec, sae_tied.W_enc.transpose(-1, -2))

def test_vanilla_sae_untied_weights(sae_params):
    sae_untied = VanillaSAE(**sae_params, tied_weights=False)
    assert sae_untied._W_dec is not None
    assert sae_untied._W_dec.shape == (sae_params['d_sae'], sae_params['d_in'])

def test_vanilla_sae_encode(vanilla_sae, sae_params):
    x = torch.randn(2, sae_params['d_in'])
    encoded = vanilla_sae.encode(x)
    assert encoded.shape == (2, sae_params['d_sae'])
    assert torch.all(encoded >= 0)  # ReLU activation

def test_vanilla_sae_decode(vanilla_sae, sae_params):
    z = torch.randn(2, sae_params['d_sae'])
    decoded = vanilla_sae.decode(z)
    assert decoded.shape == (2, sae_params['d_in'])

def test_vanilla_sae_forward(vanilla_sae, sae_params):
    x = torch.randn(2, sae_params['d_in'])
    encoded = vanilla_sae.encode(x)
    decoded = vanilla_sae.decode(encoded)
    assert decoded.shape == x.shape

@pytest.fixture
def sae_training_wrapper(vanilla_sae):
    return VanillaSAETrainingWrapper(sae=vanilla_sae, l1_coeff=0.1)

def test_vanilla_sae_training_wrapper_initialization(sae_training_wrapper, vanilla_sae):
    assert sae_training_wrapper.sae is vanilla_sae
    assert sae_training_wrapper.l1_coeff == 0.1

def test_vanilla_sae_training_wrapper_forward_pass(sae_training_wrapper, sae_params):
    x = torch.randn(2, sae_params['d_in'])
    output = sae_training_wrapper.training_forward_pass(x)
    
    assert isinstance(output, TrainStepOutput)
    assert output.sae_in.shape == x.shape
    assert output.sae_out.shape == x.shape
    assert output.sae_act.shape == (2, sae_params['d_sae'])
    assert isinstance(output.loss, torch.Tensor)
    assert output.loss.ndim == 0  # scalar
    assert 'mse_loss' in output.loss_dict
    assert 'l1_loss' in output.loss_dict

def test_vanilla_sae_training_wrapper_on_train_step_end(sae_training_wrapper):
    initial_W_dec = sae_training_wrapper.sae.W_dec.clone()
    sae_training_wrapper.on_train_step_end()
    
    if not sae_training_wrapper.sae.tied_weights:
        assert not torch.allclose(sae_training_wrapper.sae.W_dec, initial_W_dec)
        assert_close(
            sae_training_wrapper.sae.W_dec.norm(dim=-1),
            torch.ones_like(sae_training_wrapper.sae.W_dec.norm(dim=-1)),
            atol=1e-6,
            rtol=1e-3
        )
    else:
        assert_close(sae_training_wrapper.sae.W_dec, initial_W_dec)

if __name__ == "__main__":
    pytest.main()