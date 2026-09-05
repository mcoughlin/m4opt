from importlib import resources

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable, unique

from ... import missions
from .. import app
from . import data


@pytest.fixture
def fits_path():
    with resources.path(data, "800.fits") as path:
        yield str(path)


@pytest.fixture
def ecsv_path(tmp_path):
    return tmp_path / "example.ecsv"


@pytest.fixture
def gif_path(tmp_path):
    return tmp_path / "example.gif"


@pytest.fixture(params=[None, -14])
def run_scheduler(fits_path, ecsv_path, gif_path, run_cli, request):
    absmag_mean = request.param

    def func(*args):
        args = [
            *args,
            "--bandpass=NUV",
            "--nside=128",
            "--deadline=6hour",
            "--no-appmag-dist",
        ]
        if absmag_mean is not None:
            args = [*args, f"--absmag-mean={absmag_mean}"]
        result = run_cli(app, "schedule", fits_path, ecsv_path, *args)
        assert result.exit_code == 0
        table = QTable.read(ecsv_path)

        start_time_diff = table["start_time"][1:] - table["start_time"][:-1]

        assert (start_time_diff >= 0 * u.s).all(), "time intervals must be monotonic"
        assert (start_time_diff - table["duration"][:-1] >= -1e-3 * u.s).all(), (
            "time intervals must be non-overlapping"
        )

        assert (table["action"][::2] == "observe").all(), (
            "even actions must be 'observe'"
        )
        assert (table["action"][1::2] == "slew").all(), "odd actions must be 'slew'"

        observations = table[table["action"] == "observe"]
        num_fields = len(unique(observations["target_coord"].to_table()))
        num_visits = table.meta["args"]["visits"]
        assert len(observations) == num_visits * num_fields, (
            f"there are {num_fields} observations of each field"
        )

        assert (
            observations["duration"] + 1e-3 * u.s >= table.meta["args"]["exptime_min"]
        ).all()
        assert (observations["duration"] <= table.meta["args"]["exptime_max"]).all()

        result = run_cli(
            app,
            "animate",
            ecsv_path,
            gif_path,
            "--time-step=8hour",
            "--inset-center=35d -31d",
            "--inset-radius=11deg",
        )
        assert result.exit_code == 0
        assert gif_path.read_bytes().startswith(b"GIF89a")
        return table

    return func


def test_end_to_end_no_solution(run_scheduler):
    table = run_scheduler("--timelimit=1s", "--exptime-min=5hour", "--cutoff=0.1")
    assert len(table) == 0
    assert table.meta["solution_status"].startswith("aborted")
    assert table.meta["objective_value"] == pytest.approx(0, abs=1e-7)
    assert table.meta["total_time"]["slack"] == 6 * u.hour


def test_end_to_end_solution(run_scheduler):
    table = run_scheduler("--timelimit=1min", "--exptime-min=300s")
    assert len(table) >= 3


def test_field_index_identifies_the_sky_grid_row(run_scheduler):
    """Each observation names the sky grid row it points at."""
    table = run_scheduler("--timelimit=1min", "--exptime-min=300s")
    observations = table[table["action"] == "observe"]
    assert len(observations) > 0

    mission = getattr(missions, table.meta["args"]["mission"])
    grid = mission.skygrid
    if isinstance(grid, dict):
        grid = grid[table.meta["args"]["skygrid"]]
    indices = np.asarray(observations["field_index"])
    assert np.all(indices >= 0)
    separation = grid[indices].separation(observations["target_coord"])
    np.testing.assert_allclose(separation.deg, 0, atol=1e-9)

    # A slew belongs to no field.
    assert np.all(np.asarray(table[table["action"] == "slew"]["field_index"]) == -1)


def test_fixed_exptime_with_appmag_dist(fits_path, ecsv_path, run_cli):
    """Fixed exposure time mode should work when appmag_dist is True (default).

    Regression test for https://github.com/m4opt/m4opt/issues/XXX:
    When --absmag-mean is not provided (fixed exposure time) but appmag_dist
    defaults to True, the scheduler would crash with an UnboundLocalError
    accessing piecewise_breakpoints.
    """
    result = run_cli(
        app,
        "schedule",
        fits_path,
        ecsv_path,
        "--bandpass=NUV",
        "--nside=128",
        "--deadline=6hour",
        "--exptime-min=300s",
        "--timelimit=1s",
        # Notably: no --no-appmag-dist and no --absmag-mean
    )
    assert result.exit_code == 0
