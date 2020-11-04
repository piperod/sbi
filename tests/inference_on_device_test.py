# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
from torch import zeros, ones, eye
from torch.distributions import MultivariateNormal
from sbi.simulators import linear_gaussian
from sbi import utils as utils
from sbi.inference import SNPE, SNLE, SNRE
from sbi.utils.torchutils import process_device


devices = ["cpu", "cuda:0"]


@pytest.mark.slow
@pytest.mark.requires_cuda
@pytest.mark.parametrize(
    "method, model",
    [
        (SNPE, "mdn"),
        (SNPE, "maf"),
        # (SNPE, "nsf"), # Broken.
        (SNLE, "maf"),
        (SNLE, "nsf"),
        (SNRE, "mlp"),
        (SNRE, "resnet"),
    ],
)
@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_training_and_mcmc_on_device(method, model, device):
    """Test training on devices.

    This test does not check training speeds.

    """
    device = process_device(device)

    num_dim = 2
    num_samples = 10
    num_simulations = 500
    max_num_epochs = 5

    x_o = zeros(1, num_dim)
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    if method == SNPE:
        kwargs = dict(
            density_estimator=utils.posterior_nn(model=model),
            sample_with_mcmc=True,
            mcmc_method="slice_np",
        )
    elif method == SNLE:
        kwargs = dict(
            density_estimator=utils.likelihood_nn(model=model),
            mcmc_method="slice_np_vectorized",
        )
    elif method == SNRE:
        kwargs = dict(classifier=utils.classifier_nn(model=model), mcmc_method="nuts",)
    else:
        raise ValueError()

    infer = method(simulator, prior, show_progress_bars=False, device=device, **kwargs)

    # Test for two rounds.
    posterior1 = infer(
        num_simulations=num_simulations,
        training_batch_size=100,
        max_num_epochs=max_num_epochs,
    ).set_default_x(x_o)

    posterior2 = infer(
        num_simulations=num_simulations,
        training_batch_size=100,
        max_num_epochs=max_num_epochs,
        proposal=posterior1,
    ).set_default_x(x_o)

    posterior2.sample((num_samples,), show_progress_bars=False)


@pytest.mark.parametrize("device", ["cpu", "gpu", "cuda", "cuda:0", "cuda:42"])
def test_process_device(device: str):
    process_device(device)
