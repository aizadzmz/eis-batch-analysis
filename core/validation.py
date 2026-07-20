#KK and Z-HT
from typing import Union

from pyimpspec import (
    KramersKronigResult,
    ZHITResult,
    perform_kramers_kronig_test,
    perform_zhit,
)

ValidationResult = Union[KramersKronigResult, ZHITResult]


def run_kk_test(dataset, **kwargs) -> KramersKronigResult:
    """
    Run pyimpspec's linear Kramers-Kronig test on the dataset's currently
    unmasked points. Extra keyword arguments are passed through to
    pyimpspec.perform_kramers_kronig_test (e.g. test, num_RC, admittance).
    """
    return perform_kramers_kronig_test(dataset.data, **kwargs)


def run_zhit(dataset, **kwargs) -> ZHITResult:
    """
    Run pyimpspec's Z-HIT analysis on the dataset's currently unmasked
    points. The modulus is reconstructed from the phase data, which helps
    detect non-steady-state artifacts such as low-frequency drift. Extra
    keyword arguments are passed through to pyimpspec.perform_zhit
    (e.g. smoothing, interpolation, window).
    """
    return perform_zhit(dataset.data, **kwargs)


def mask_residual_outliers(
    dataset, result: ValidationResult, threshold_percent: float
) -> None:
    """
    Mask points whose relative residual (real or imaginary, in percent)
    exceeds threshold_percent, in place. Points already masked stay masked.
    Works with both Kramers-Kronig and Z-HIT results.

    result must have been produced by run_kk_test/run_zhit on this dataset
    without changing the dataset's mask in between, since residuals are only
    reported for the points that were unmasked at test time.
    """
    mask = dataset.data.get_mask()
    unmasked_indices = [
        i
        for i in range(dataset.data.get_num_points(masked=None))
        if not mask.get(i, False)
    ]

    _, res_re, res_im = result.get_residuals_data()
    if len(unmasked_indices) != len(res_re):
        raise ValueError(
            "Validation result does not match the dataset's current mask; "
            "re-run the validation first."
        )

    for idx, re, im in zip(unmasked_indices, res_re, res_im):
        if abs(re) > threshold_percent or abs(im) > threshold_percent:
            mask[idx] = True

    dataset.data.set_mask(mask)


# Backwards-compatible alias (residual masking is method-agnostic).
mask_kk_outliers = mask_residual_outliers
