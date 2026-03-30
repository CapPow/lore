"""
tests/test_features.py

Unit tests for pure functions in lore/features.py.
No raster files are read; raster-sampling functions are not tested here
as they require real file fixtures (integration concern).

Run with:
    pytest tests/test_features.py -v
"""

import math

import numpy as np
import pandas as pd
import pytest

from lore.features import (
    UNKNOWN_TOKEN,
    _clean_verbatim_name,
    _cyclical_doy,
    _fit_minmax,
    _minmax_norm,
    apply_name_encoder,
    fit_name_encoder,
)


# ---------------------------------------------------------------------------
# _minmax_norm / _fit_minmax
# ---------------------------------------------------------------------------

class TestMinmaxNorm:

    def test_normalises_to_unit_range(self):
        arr = np.array([0.0, 5.0, 10.0], dtype=np.float32)
        vmin, vmax = _fit_minmax(arr)
        result = _minmax_norm(arr, vmin, vmax)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    def test_midpoint_is_half(self):
        arr = np.array([0.0, 50.0, 100.0], dtype=np.float32)
        vmin, vmax = _fit_minmax(arr)
        result = _minmax_norm(arr, vmin, vmax)
        assert result[1] == pytest.approx(0.5)

    def test_zero_range_returns_zeros(self):
        arr = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        result = _minmax_norm(arr, 5.0, 5.0)
        assert (result == 0.0).all()

    def test_nan_propagated(self):
        arr = np.array([0.0, np.nan, 10.0], dtype=np.float32)
        result = _minmax_norm(arr, 0.0, 10.0)
        assert np.isnan(result[1])
        assert not np.isnan(result[0])
        assert not np.isnan(result[2])

    def test_fit_minmax_ignores_nan(self):
        arr = np.array([1.0, np.nan, 5.0], dtype=np.float32)
        vmin, vmax = _fit_minmax(arr)
        assert vmin == pytest.approx(1.0)
        assert vmax == pytest.approx(5.0)

    def test_fit_minmax_all_nan_returns_defaults(self):
        arr = np.array([np.nan, np.nan], dtype=np.float32)
        vmin, vmax = _fit_minmax(arr)
        assert vmin == 0.0
        assert vmax == 1.0

    def test_output_dtype_is_float32(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = _minmax_norm(arr, 1.0, 3.0)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# _clean_verbatim_name
# ---------------------------------------------------------------------------

class TestCleanVerbatimName:

    def test_passes_clean_binomial(self):
        assert _clean_verbatim_name("Peromyscus maniculatus") == "Peromyscus maniculatus"

    def test_strips_qualifier_terms(self):
        assert _clean_verbatim_name("Peromyscus cf. maniculatus") == "Peromyscus maniculatus"
        assert _clean_verbatim_name("Peromyscus sp.") == "Peromyscus"
        assert _clean_verbatim_name("Peromyscus aff. sonoriensis") == "Peromyscus sonoriensis"

    def test_strips_multiple_qualifiers(self):
        result = _clean_verbatim_name("Peromyscus ssp. maniculatus subsp. sonoriensis")
        assert "ssp." not in result
        assert "subsp." not in result

    def test_returns_empty_string_for_none(self):
        assert _clean_verbatim_name(None) == ""

    def test_returns_empty_string_for_empty_string(self):
        assert _clean_verbatim_name("") == ""

    def test_returns_empty_string_for_whitespace(self):
        assert _clean_verbatim_name("   ") == ""

    def test_case_insensitive_qualifier_removal(self):
        # qualifier terms in _OMIT_TERMS are lowercase; input may have uppercase
        result = _clean_verbatim_name("Peromyscus CF. maniculatus")
        assert "CF." not in result


# ---------------------------------------------------------------------------
# fit_name_encoder / apply_name_encoder
# ---------------------------------------------------------------------------

class TestNameEncoder:

    def test_unknown_token_is_index_zero(self):
        names = pd.Series(["Taxon A", "Taxon B"])
        encoder = fit_name_encoder(names)
        assert encoder[UNKNOWN_TOKEN] == 0

    def test_known_names_start_at_one(self):
        names = pd.Series(["Taxon A", "Taxon B"])
        encoder = fit_name_encoder(names)
        assert min(v for k, v in encoder.items() if k != UNKNOWN_TOKEN) == 1

    def test_encoder_is_deterministic(self):
        names = pd.Series(["Taxon B", "Taxon A", "Taxon C"])
        enc1 = fit_name_encoder(names)
        enc2 = fit_name_encoder(names)
        assert enc1 == enc2

    def test_encoder_vocabulary_sorted(self):
        names = pd.Series(["Taxon C", "Taxon A", "Taxon B"])
        encoder = fit_name_encoder(names)
        # Remove unknown token and check remaining indices are in sorted order
        taxa = {k: v for k, v in encoder.items() if k != UNKNOWN_TOKEN}
        sorted_names = sorted(taxa.keys())
        assert [taxa[n] for n in sorted_names] == list(range(1, len(taxa) + 1))

    def test_apply_encodes_known_names(self):
        names = pd.Series(["Taxon A", "Taxon B"])
        encoder = fit_name_encoder(names)
        encoded = apply_name_encoder(names, encoder)
        assert encoded.dtype == np.int32
        assert all(v > 0 for v in encoded)

    def test_apply_encodes_unknown_as_zero(self):
        names = pd.Series(["Taxon A"])
        encoder = fit_name_encoder(names)
        unseen = pd.Series(["Taxon Z"])
        encoded = apply_name_encoder(unseen, encoder)
        assert encoded[0] == 0

    def test_empty_and_null_names_excluded_from_vocab(self):
        names = pd.Series(["Taxon A", "", None, "Taxon B"])
        encoder = fit_name_encoder(names)
        assert "" not in encoder
        assert None not in encoder

    def test_single_taxon_produces_vocab_size_two(self):
        # vocab = {UNKNOWN_TOKEN: 0, "Taxon A": 1} -- n_vocab = 2 for single source
        names = pd.Series(["Taxon A", "Taxon A", "Taxon A"])
        encoder = fit_name_encoder(names)
        assert len(encoder) == 2


# ---------------------------------------------------------------------------
# _cyclical_doy
# ---------------------------------------------------------------------------

class TestCyclicalDoy:

    def test_known_date_values(self):
        # Jan 1 = doy 1; angle = 2π/365
        dates = pd.Series(["2020-01-01"])
        sin_doy, cos_doy = _cyclical_doy(dates)
        expected_angle = 2.0 * math.pi * 1.0 / 365.0
        assert sin_doy[0] == pytest.approx(math.sin(expected_angle), abs=1e-5)
        assert cos_doy[0] == pytest.approx(math.cos(expected_angle), abs=1e-5)

    def test_missing_date_produces_zeros(self):
        dates = pd.Series([None])
        sin_doy, cos_doy = _cyclical_doy(dates)
        assert sin_doy[0] == pytest.approx(0.0)
        assert cos_doy[0] == pytest.approx(0.0)

    def test_unparseable_date_produces_zeros(self):
        dates = pd.Series(["not_a_date"])
        sin_doy, cos_doy = _cyclical_doy(dates)
        assert sin_doy[0] == pytest.approx(0.0)
        assert cos_doy[0] == pytest.approx(0.0)

    def test_output_dtype_is_float32(self):
        dates = pd.Series(["2020-06-15"])
        sin_doy, cos_doy = _cyclical_doy(dates)
        assert sin_doy.dtype == np.float32
        assert cos_doy.dtype == np.float32

    def test_sin_cos_satisfy_pythagorean_identity(self):
        dates = pd.Series(["2020-03-15", "2020-07-04", "2020-11-30"])
        sin_doy, cos_doy = _cyclical_doy(dates)
        for s, c in zip(sin_doy, cos_doy):
            assert s**2 + c**2 == pytest.approx(1.0, abs=1e-5)

    def test_mixed_valid_and_missing(self):
        dates = pd.Series(["2020-06-15", None, "2020-12-01"])
        sin_doy, cos_doy = _cyclical_doy(dates)
        assert not math.isnan(sin_doy[0])
        assert sin_doy[1] == pytest.approx(0.0)
        assert cos_doy[1] == pytest.approx(0.0)
        assert not math.isnan(sin_doy[2])
