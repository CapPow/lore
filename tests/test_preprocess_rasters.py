"""
tests/test_preprocess_rasters.py

Unit tests for helper functions in scripts/preprocess_rasters.py.
No real rasters are read. Bbox computation is tested via a synthetic
GeoDataFrame written to a temp parquet. _validate_bbox is tested with
a synthetic run_dir.

Run with:
    pytest tests/test_preprocess_rasters.py -v
"""

import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

# preprocess_rasters lives in scripts/, not a package, so we import via path
import importlib.util
import sys

def _import_preprocess():
    here    = Path(__file__).parent
    script  = here.parent / "scripts" / "preprocess_rasters.py"
    spec    = importlib.util.spec_from_file_location("preprocess_rasters", script)
    module  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

preprocess = _import_preprocess()

_get_occurrence_bbox = preprocess._get_occurrence_bbox
_validate_bbox       = preprocess._validate_bbox
_write_bbox          = preprocess._write_bbox


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_occ_parquet(tmp_path: Path, points: list[tuple[float, float]]) -> Path:
    """
    Write a minimal GeoDataFrame parquet with the given (lon, lat) points.
    Returns the parquet path.
    """
    gdf = gpd.GeoDataFrame(
        {"id": range(len(points))},
        geometry=[Point(lon, lat) for lon, lat in points],
        crs="EPSG:4326",
    )
    path = tmp_path / "occ.parquet"
    gdf.to_parquet(path)
    return path


# ---------------------------------------------------------------------------
# _get_occurrence_bbox
# ---------------------------------------------------------------------------

class TestGetOccurrenceBbox:

    def test_basic_bbox_with_buffer(self, tmp_path):
        path = _make_occ_parquet(tmp_path, [(-105.0, 35.0), (-100.0, 40.0)])
        minx, miny, maxx, maxy = _get_occurrence_bbox(path, buffer_deg=1.0)
        assert minx == pytest.approx(-106.0)
        assert miny == pytest.approx(34.0)
        assert maxx == pytest.approx(-99.0)
        assert maxy == pytest.approx(41.0)

    def test_zero_buffer(self, tmp_path):
        path = _make_occ_parquet(tmp_path, [(-105.0, 35.0), (-100.0, 40.0)])
        minx, miny, maxx, maxy = _get_occurrence_bbox(path, buffer_deg=0.0)
        assert minx == pytest.approx(-105.0)
        assert miny == pytest.approx(35.0)
        assert maxx == pytest.approx(-100.0)
        assert maxy == pytest.approx(40.0)

    def test_clamps_to_valid_lon_range(self, tmp_path):
        # Point near antimeridian; buffer should not exceed ±180
        path = _make_occ_parquet(tmp_path, [(179.5, 0.0)])
        _, _, maxx, _ = _get_occurrence_bbox(path, buffer_deg=2.0)
        assert maxx <= 180.0

    def test_clamps_to_valid_lat_range(self, tmp_path):
        # Point near north pole; buffer should not exceed ±90
        path = _make_occ_parquet(tmp_path, [(0.0, 89.5)])
        _, _, _, maxy = _get_occurrence_bbox(path, buffer_deg=2.0)
        assert maxy <= 90.0

    def test_clamps_southern_lat(self, tmp_path):
        path = _make_occ_parquet(tmp_path, [(0.0, -89.5)])
        _, miny, _, _ = _get_occurrence_bbox(path, buffer_deg=2.0)
        assert miny >= -90.0

    def test_single_point_bbox(self, tmp_path):
        path = _make_occ_parquet(tmp_path, [(-105.0, 35.0)])
        minx, miny, maxx, maxy = _get_occurrence_bbox(path, buffer_deg=1.0)
        # For a single point, min==max before buffer
        assert minx == pytest.approx(-106.0)
        assert maxx == pytest.approx(-104.0)

    def test_returns_plain_python_floats(self, tmp_path):
        path = _make_occ_parquet(tmp_path, [(-105.0, 35.0)])
        result = _get_occurrence_bbox(path, buffer_deg=1.0)
        for v in result:
            assert isinstance(v, float), f"Expected float, got {type(v)}"


# ---------------------------------------------------------------------------
# _validate_bbox
# ---------------------------------------------------------------------------

class TestValidateBbox:

    def _write_stored_bbox(self, run_dir: Path, bbox: tuple) -> None:
        _write_bbox(run_dir, bbox)

    def test_no_warning_when_bbox_matches(self, tmp_path):
        bbox = (-106.0, 34.0, -99.0, 41.0)
        self._write_stored_bbox(tmp_path, bbox)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_bbox(tmp_path, bbox, force=False)  # should not warn

    def test_warns_when_bbox_differs(self, tmp_path):
        stored = (-106.0, 34.0, -99.0, 41.0)
        current = (-110.0, 30.0, -95.0, 45.0)
        self._write_stored_bbox(tmp_path, stored)
        with pytest.warns(UserWarning, match="differs from current bbox"):
            _validate_bbox(tmp_path, current, force=False)

    def test_no_warning_when_force_true(self, tmp_path):
        stored = (-106.0, 34.0, -99.0, 41.0)
        current = (-110.0, 30.0, -95.0, 45.0)
        self._write_stored_bbox(tmp_path, stored)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_bbox(tmp_path, current, force=True)  # force skips check

    def test_no_warning_when_no_stored_bbox(self, tmp_path):
        bbox = (-106.0, 34.0, -99.0, 41.0)
        # No bbox.json written -- should be a no-op
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_bbox(tmp_path, bbox, force=False)

    def test_stale_tif_count_in_warning(self, tmp_path):
        stored = (-106.0, 34.0, -99.0, 41.0)
        current = (-110.0, 30.0, -95.0, 45.0)
        self._write_stored_bbox(tmp_path, stored)
        # Create a fake tif to verify the count appears in the warning
        (tmp_path / "fake.tif").write_bytes(b"")
        with pytest.warns(UserWarning, match="1 tif"):
            _validate_bbox(tmp_path, current, force=False)

    def test_no_warning_for_small_bbox_difference_within_tolerance(self, tmp_path):
        stored  = (-106.0, 34.0, -99.0, 41.0)
        current = (-106.005, 34.005, -99.005, 41.005)  # < 0.01 deg difference
        self._write_stored_bbox(tmp_path, stored)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_bbox(tmp_path, current, force=False)

    def test_warning_for_difference_exceeding_tolerance(self, tmp_path):
        stored  = (-106.0, 34.0, -99.0, 41.0)
        current = (-106.02, 34.0, -99.0, 41.0)  # 0.02 deg > 0.01 tolerance
        self._write_stored_bbox(tmp_path, stored)
        with pytest.warns(UserWarning, match="differs from current bbox"):
            _validate_bbox(tmp_path, current, force=False)


# ---------------------------------------------------------------------------
# _write_bbox
# ---------------------------------------------------------------------------

class TestWriteBbox:

    def test_written_file_is_valid_json(self, tmp_path):
        bbox = (-106.0, 34.0, -99.0, 41.0)
        _write_bbox(tmp_path, bbox)
        content = json.loads((tmp_path / "bbox.json").read_text())
        assert "bbox" in content
        assert content["bbox"] == list(bbox)

    def test_written_bbox_is_list_of_floats(self, tmp_path):
        bbox = (-106.0, 34.0, -99.0, 41.0)
        _write_bbox(tmp_path, bbox)
        content = json.loads((tmp_path / "bbox.json").read_text())
        assert all(isinstance(v, float) for v in content["bbox"])

    def test_roundtrip_preserves_values(self, tmp_path):
        bbox = (-106.123, 34.456, -99.789, 41.012)
        _write_bbox(tmp_path, bbox)
        stored = tuple(json.loads((tmp_path / "bbox.json").read_text())["bbox"])
        for a, b in zip(bbox, stored):
            assert a == pytest.approx(b)
