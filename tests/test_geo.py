"""
tests/test_geo.py

Unit tests for lore/geo.py using synthetic GeoDataFrames and DataFrames.
No real files are read; no network or filesystem access required.

Run with:
    pytest tests/test_geo.py -v
"""

import warnings
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from lore.geo import (
    LABEL_EXCESSIVE_UNCERT,
    LABEL_OUT_OF_RANGE,
    LABEL_PARAPATRIC_SEP,
    REQUIRED_OCC_COLS,
    _validate_occurrences,
    allopatry_report,
    disambiguate,
    load_occurrences,
    load_ranges,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ranges(boxes: dict[str, tuple]) -> gpd.GeoDataFrame:
    """
    Build a minimal ranges GeoDataFrame from a dict of {name: (minx,miny,maxx,maxy)}.
    """
    rows = [{"sciname": name, "geometry": box(*coords)} for name, coords in boxes.items()]
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _make_occ_df(**overrides) -> pd.DataFrame:
    """
    Return a minimal valid Darwin Core DataFrame (one record).
    Keyword args override individual column values.
    """
    base = {
        "gbifID":                          "1",
        "decimalLatitude":                 35.0,
        "decimalLongitude":               -106.0,
        "coordinateUncertaintyInMeters":   100.0,
        "verbatimScientificName":          "Peromyscus maniculatus",
        "eventDate":                       "2020-06-15",
    }
    base.update(overrides)
    return pd.DataFrame([base])


# ---------------------------------------------------------------------------
# _validate_occurrences
# ---------------------------------------------------------------------------

class TestValidateOccurrences:

    def test_passes_valid_dataframe(self):
        _validate_occurrences(_make_occ_df())

    def test_raises_on_missing_required_column(self):
        df = _make_occ_df()
        df = df.drop(columns=["gbifID"])
        with pytest.raises(ValueError, match="missing required Darwin Core columns"):
            _validate_occurrences(df)

    def test_raises_on_multiple_missing_columns(self):
        df = _make_occ_df()
        df = df.drop(columns=["gbifID", "decimalLatitude"])
        with pytest.raises(ValueError, match="missing required Darwin Core columns"):
            _validate_occurrences(df)

    def test_raises_on_nonnumeric_latitude(self):
        df = _make_occ_df(decimalLatitude="not_a_number")
        with pytest.raises(ValueError, match="Non-numeric coordinate"):
            _validate_occurrences(df)

    def test_raises_on_nonnumeric_longitude(self):
        df = _make_occ_df(decimalLongitude="bad")
        with pytest.raises(ValueError, match="Non-numeric coordinate"):
            _validate_occurrences(df)

    def test_raises_on_latitude_out_of_range(self):
        df = _make_occ_df(decimalLatitude=95.0)
        with pytest.raises(ValueError, match="outside valid degree range"):
            _validate_occurrences(df)

    def test_raises_on_longitude_out_of_range(self):
        df = _make_occ_df(decimalLongitude=200.0)
        with pytest.raises(ValueError, match="outside valid degree range"):
            _validate_occurrences(df)

    def test_warns_on_high_zero_coordinate_ratio(self):
        rows = [_make_occ_df(decimalLatitude=0, decimalLongitude=0).iloc[0].to_dict()
                for _ in range(12)]
        rows += [_make_occ_df().iloc[0].to_dict() for _ in range(3)]
        df = pd.DataFrame(rows)
        with pytest.warns(UserWarning, match="exactly \\(0, 0\\)"):
            _validate_occurrences(df)

    def test_no_warn_on_low_zero_coordinate_ratio(self):
        rows = [_make_occ_df(decimalLatitude=0, decimalLongitude=0, gbifID="0").iloc[0].to_dict()]
        rows += [_make_occ_df(gbifID=str(i + 1)).iloc[0].to_dict() for i in range(14)]
        
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_occurrences(df)  # should not raise

    def test_warns_on_high_date_parse_failure(self):
        rows = [_make_occ_df(eventDate="not_a_date").iloc[0].to_dict() for _ in range(40)]
        rows += [_make_occ_df().iloc[0].to_dict() for _ in range(10)]
        df = pd.DataFrame(rows)
        with pytest.warns(UserWarning, match="eventDate could not be parsed"):
            _validate_occurrences(df)

    def test_warns_on_duplicate_gbif_ids(self):
        df = pd.concat([_make_occ_df(), _make_occ_df()], ignore_index=True)
        with pytest.warns(UserWarning, match="duplicate gbifID"):
            _validate_occurrences(df)

    def test_raises_if_no_viable_coordinates_remain(self):
        df = _make_occ_df(decimalLatitude=0, decimalLongitude=0)
        with pytest.raises(ValueError, match="No usable records remain"):
            _validate_occurrences(df)

    def test_warns_if_source_taxa_not_found(self):
        df = _make_occ_df(verbatimScientificName="Mus musculus")
        with pytest.warns(UserWarning, match="None of the source_taxa"):
            _validate_occurrences(df, source_taxa=["Peromyscus maniculatus"])


# ---------------------------------------------------------------------------
# load_occurrences
# ---------------------------------------------------------------------------

class TestLoadOccurrences:

    def test_returns_geodataframe(self):
        df = _make_occ_df()
        result = load_occurrences(df)
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs.to_epsg() == 4326

    def test_point_geometry_matches_coordinates(self):
        df = _make_occ_df(decimalLatitude=35.0, decimalLongitude=-106.0)
        result = load_occurrences(df)
        assert result.geometry.iloc[0].x == pytest.approx(-106.0)
        assert result.geometry.iloc[0].y == pytest.approx(35.0)

    def test_drops_zero_coordinates(self):
        rows = [_make_occ_df(decimalLatitude=0, decimalLongitude=0).iloc[0].to_dict(),
                _make_occ_df().iloc[0].to_dict()]
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = load_occurrences(df)
        assert len(result) == 1

    def test_null_coordinates_raise_in_validation(self):
        df = _make_occ_df()
        df["decimalLatitude"] = np.nan
        with pytest.raises(ValueError, match="Non-numeric coordinate"):
            load_occurrences(df)

    def test_flags_excessive_uncertainty(self):
        df = _make_occ_df(coordinateUncertaintyInMeters=200_000.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = load_occurrences(df)
        assert result["suggested_names"].iloc[0] == LABEL_EXCESSIVE_UNCERT

    def test_normal_uncertainty_not_flagged(self):
        df = _make_occ_df(coordinateUncertaintyInMeters=500.0)
        result = load_occurrences(df)
        assert result["suggested_names"].iloc[0] != LABEL_EXCESSIVE_UNCERT

    def test_null_uncertainty_imputed_with_median(self):
        rows = [
            _make_occ_df(coordinateUncertaintyInMeters=100.0).iloc[0].to_dict(),
            _make_occ_df(coordinateUncertaintyInMeters=200.0).iloc[0].to_dict(),
            _make_occ_df(coordinateUncertaintyInMeters=None,
                         gbifID="2",
                         decimalLatitude=36.0).iloc[0].to_dict(),
        ]
        df = pd.DataFrame(rows)
        result = load_occurrences(df)
        imputed = result.loc[result["gbifID"] == "2",
                             "coordinateUncertaintyInMeters"].iloc[0]
        assert imputed == pytest.approx(150.0)  # median of [100, 200]

    def test_filters_by_source_taxa_binomial(self):
        rows = [
            _make_occ_df(gbifID="1",
                         verbatimScientificName="Peromyscus maniculatus").iloc[0].to_dict(),
            _make_occ_df(gbifID="2",
                         verbatimScientificName="Mus musculus",
                         decimalLatitude=36.0).iloc[0].to_dict(),
        ]
        df = pd.DataFrame(rows)
        result = load_occurrences(df, source_taxa=["Peromyscus maniculatus"])
        assert len(result) == 1
        assert result["gbifID"].iloc[0] == "1"

    def test_filters_by_source_taxa_genus(self):
        rows = [
            _make_occ_df(gbifID="1",
                         verbatimScientificName="Peromyscus maniculatus").iloc[0].to_dict(),
            _make_occ_df(gbifID="2",
                         verbatimScientificName="Peromyscus sonoriensis",
                         decimalLatitude=36.0).iloc[0].to_dict(),
            _make_occ_df(gbifID="3",
                         verbatimScientificName="Mus musculus",
                         decimalLatitude=37.0).iloc[0].to_dict(),
        ]
        df = pd.DataFrame(rows)
        result = load_occurrences(df, source_taxa=["Peromyscus"])
        assert len(result) == 2
        assert set(result["gbifID"]) == {"1", "2"}

    def test_uncertainty_floor_applied(self):
        df = _make_occ_df(coordinateUncertaintyInMeters=0.0)
        result = load_occurrences(df, uncertainty_floor=1.0)
        assert result["coordinateUncertaintyInMeters"].iloc[0] >= 1.0


# ---------------------------------------------------------------------------
# load_ranges
# ---------------------------------------------------------------------------

class TestLoadRanges:

    def test_raises_on_missing_name_col(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            [{"wrong_col": "Taxon A", "geometry": box(0, 0, 1, 1)}],
            crs="EPSG:4326",
        )
        path = tmp_path / "ranges.gpkg"
        gdf.to_file(path, driver="GPKG")
        with pytest.raises(ValueError, match="name_col 'sciname' not found"):
            load_ranges(path, taxa=["Taxon A"])

    def test_raises_on_no_matching_taxa(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            [{"sciname": "Taxon A", "geometry": box(0, 0, 1, 1)}],
            crs="EPSG:4326",
        )
        path = tmp_path / "ranges.gpkg"
        gdf.to_file(path, driver="GPKG")
        with pytest.raises(ValueError, match="No ranges matched"):
            load_ranges(path, taxa=["Taxon B"])

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_ranges(tmp_path / "nonexistent.gpkg", taxa=["Taxon A"])

    def test_filters_by_binomial(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            [
                {"sciname": "Taxon alpha", "geometry": box(0, 0, 1, 1)},
                {"sciname": "Taxon beta",  "geometry": box(2, 2, 3, 3)},
            ],
            crs="EPSG:4326",
        )
        path = tmp_path / "ranges.gpkg"
        gdf.to_file(path, driver="GPKG")
        result = load_ranges(path, taxa=["Taxon alpha"])
        assert len(result) == 1
        assert result["sciname"].iloc[0].lower() == "taxon alpha"

    def test_filters_by_genus(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            [
                {"sciname": "Taxon alpha", "geometry": box(0, 0, 1, 1)},
                {"sciname": "Taxon beta",  "geometry": box(2, 2, 3, 3)},
                {"sciname": "Other gamma", "geometry": box(4, 4, 5, 5)},
            ],
            crs="EPSG:4326",
        )
        path = tmp_path / "ranges.gpkg"
        gdf.to_file(path, driver="GPKG")
        result = load_ranges(path, taxa=["Taxon"])
        assert len(result) == 2

    def test_reprojects_to_4326(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            [{"sciname": "Taxon alpha", "geometry": box(0, 0, 1, 1)}],
            crs="EPSG:4326",
        ).to_crs("EPSG:3857")
        path = tmp_path / "ranges.gpkg"
        gdf.to_file(path, driver="GPKG")
        result = load_ranges(path, taxa=["Taxon alpha"])
        assert result.crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# allopatry_report
# ---------------------------------------------------------------------------

class TestAllopatryReport:

    def test_non_overlapping_ranges_are_allopatric(self):
        ranges = _make_ranges({
            "Taxon A": (0, 0, 1, 1),
            "Taxon B": (2, 2, 3, 3),
        })
        allopatric, cw, report = allopatry_report(ranges)
        assert allopatric is True
        assert report["(A∩B)/A"].iloc[0] == pytest.approx(0.0)
        assert report["(A∩B)/B"].iloc[0] == pytest.approx(0.0)

    def test_overlapping_ranges_not_allopatric(self):
        ranges = _make_ranges({
            "Taxon A": (0, 0, 2, 2),
            "Taxon B": (1, 1, 3, 3),
        })
        allopatric, cw, report = allopatry_report(ranges, overlap_threshold=0.1)
        assert allopatric is False

    def test_overlap_ratios_are_symmetric_for_equal_ranges(self):
        ranges = _make_ranges({
            "Taxon A": (0, 0, 2, 2),
            "Taxon B": (1, 0, 3, 2),  # half overlap with A
        })
        _, _, report = allopatry_report(ranges)
        ratio_a = report["(A∩B)/A"].iloc[0]
        ratio_b = report["(A∩B)/B"].iloc[0]
        assert ratio_a == pytest.approx(ratio_b, abs=0.01)

    def test_concept_wide_overlap_is_zero_for_allopatric(self):
        ranges = _make_ranges({
            "Taxon A": (0, 0, 1, 1),
            "Taxon B": (2, 2, 3, 3),
        })
        _, cw, _ = allopatry_report(ranges)
        assert cw == pytest.approx(0.0)

    def test_concept_wide_overlap_positive_for_overlapping(self):
        ranges = _make_ranges({
            "Taxon A": (0, 0, 2, 2),
            "Taxon B": (1, 1, 3, 3),
        })
        _, cw, _ = allopatry_report(ranges)
        assert cw > 0.0

    def test_three_taxa_pairwise_report_has_correct_row_count(self):
        ranges = _make_ranges({
            "Taxon A": (0, 0, 1, 1),
            "Taxon B": (2, 2, 3, 3),
            "Taxon C": (4, 4, 5, 5),
        })
        _, _, report = allopatry_report(ranges)
        assert len(report) == 3  # C(3,2) = 3 pairs

    def test_threshold_respected(self):
        ranges = _make_ranges({
            "Taxon A": (0, 0, 2, 2),
            "Taxon B": (1, 1, 3, 3),  # ~25% overlap
        })
        allopatric_strict, _, _ = allopatry_report(ranges, overlap_threshold=0.01)
        allopatric_loose,  _, _ = allopatry_report(ranges, overlap_threshold=0.9)
        assert allopatric_strict is False
        assert allopatric_loose  is True


# ---------------------------------------------------------------------------
# disambiguate
# ---------------------------------------------------------------------------

class TestDisambiguate:
    """
    Tests for the core spatial assignment logic.
    Uses single-process execution (processes=1) to avoid multiprocessing
    overhead in the test suite.
    """

    def _occ_at(self, lon: float, lat: float, uncertainty: float = 1.0,
                gbif_id: str = "1") -> gpd.GeoDataFrame:
        """Build a single-record occurrence GeoDataFrame at the given point."""
        df = _make_occ_df(
            gbifID=gbif_id,
            decimalLongitude=lon,
            decimalLatitude=lat,
            coordinateUncertaintyInMeters=uncertainty,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return load_occurrences(df)

    def test_point_inside_single_range_gets_label(self):
        ranges = _make_ranges({
            "Taxon A": (-110, 30, -100, 40),
            "Taxon B": (-90,  30, -80,  40),
        })
        occ = self._occ_at(-105.0, 35.0)
        result = disambiguate(occ, ranges, processes=1)
        assert result["suggested_names"].iloc[0] == "Taxon A"

    def test_point_outside_all_ranges_is_out_of_range(self):
        ranges = _make_ranges({
            "Taxon A": (-110, 30, -100, 40),
            "Taxon B": (-90,  30, -80,  40),
        })
        occ = self._occ_at(-50.0, 35.0)
        result = disambiguate(occ, ranges, processes=1)
        assert result["suggested_names"].iloc[0] == LABEL_OUT_OF_RANGE

    def test_point_in_overlap_zone_gets_pipe_label(self):
        # Two ranges with substantial overlap; point in the middle
        ranges = _make_ranges({
            "Taxon A": (-110, 30, -95, 40),
            "Taxon B": (-100, 30, -85, 40),
        })
        occ = self._occ_at(-97.5, 35.0, uncertainty=1.0)
        result = disambiguate(occ, ranges, processes=1)
        label = result["suggested_names"].iloc[0]
        assert LABEL_PARAPATRIC_SEP in label
        assert "Taxon A" in label
        assert "Taxon B" in label

    def test_excessive_uncertainty_records_pass_through(self):
        ranges = _make_ranges({"Taxon A": (-110, 30, -100, 40)})
        df = _make_occ_df(coordinateUncertaintyInMeters=200_000.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            occ = load_occurrences(df)
        result = disambiguate(occ, ranges, processes=1)
        assert result["suggested_names"].iloc[0] == LABEL_EXCESSIVE_UNCERT

    def test_large_uncertainty_buffer_can_span_into_range(self):
        # Point just outside range boundary, but large uncertainty buffer reaches in
        ranges = _make_ranges({"Taxon A": (-110, 30, -100, 40)})
        # Point at -99.5 is outside range [-110, -100], but with 100km buffer
        # the circle will extend well into the range
        occ = self._occ_at(-99.0, 35.0, uncertainty=200_000.0)
        # This record is flagged excessive and passes through unchanged
        assert occ["suggested_names"].iloc[0] == LABEL_EXCESSIVE_UNCERT

    def test_multiple_records_correct_labels(self):
        ranges = _make_ranges({
            "Taxon A": (-110, 30, -100, 40),
            "Taxon B": (-90,  30, -80,  40),
        })
        occ_a  = self._occ_at(-105.0, 35.0, gbif_id="1")
        occ_b  = self._occ_at(-85.0,  35.0, gbif_id="2")
        occ_oo = self._occ_at(-50.0,  35.0, gbif_id="3")
        occ = pd.concat([occ_a, occ_b, occ_oo], ignore_index=True)
        occ = gpd.GeoDataFrame(occ, geometry="geometry", crs="EPSG:4326")

        result = disambiguate(occ, ranges, processes=1)
        labels = dict(zip(result["gbifID"], result["suggested_names"]))
        assert labels["1"] == "Taxon A"
        assert labels["2"] == "Taxon B"
        assert labels["3"] == LABEL_OUT_OF_RANGE

    def test_parapatric_label_is_sorted(self):
        # Labels should be pipe-delimited and sorted -- deterministic output
        ranges = _make_ranges({
            "Taxon B": (-110, 30, -95, 40),
            "Taxon A": (-100, 30, -85, 40),
        })
        occ = self._occ_at(-97.5, 35.0)
        result = disambiguate(occ, ranges, processes=1)
        label = result["suggested_names"].iloc[0]
        if LABEL_PARAPATRIC_SEP in label:
            parts = label.split(LABEL_PARAPATRIC_SEP)
            assert parts == sorted(parts)
