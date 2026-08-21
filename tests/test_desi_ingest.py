from __future__ import annotations

import numpy as np
import pytest

from p3sf.ingest.desi import (
    ScienceRecord,
    count_tolerance_duplicates,
    exposure_summary,
    merge_cameras_official,
    parse_catalog_bool,
)


@pytest.mark.parametrize("value", ["t", "true", "1", " T ", True, 1])
def test_parse_catalog_bool_true(value: object) -> None:
    assert parse_catalog_bool(value) is True


@pytest.mark.parametrize("value", ["f", "false", "0", " F ", False, 0])
def test_parse_catalog_bool_false_including_prior_exact_f_defect(value: object) -> None:
    assert parse_catalog_bool(value) is False


@pytest.mark.parametrize("value", ["", "yes", "no", 2, None, np.nan])
def test_parse_catalog_bool_fails_closed(value: object) -> None:
    with pytest.raises(ValueError, match="unknown catalog boolean"):
        parse_catalog_bool(value)


def _record(**changes: object) -> ScienceRecord:
    values: dict[str, object] = {
        "object_id": "SDSSJ161711.42+063833.4",
        "specprod": "iron",
        "survey": "main",
        "program": "bright",
        "healpix": 31535,
        "targetid": 39627951624425454,
        "zpix_id": "158456325139248767580969635822",
    }
    values.update(changes)
    return ScienceRecord(**values)  # type: ignore[arg-type]


def test_full_source_identity_prevents_j161_program_collision() -> None:
    bright = _record()
    dark = _record(program="dark", zpix_id="237684487653513105174513586158")

    assert bright.spectrum_id != dark.spectrum_id
    assert bright.filename != dark.filename
    assert "bright" in bright.filename
    assert "dark" in dark.filename
    assert "tid39627951624425454" in bright.filename


def test_survey_and_source_product_are_identity_coordinates() -> None:
    baseline = _record()
    records = [
        baseline,
        _record(survey="sv3"),
        _record(specprod="guadalupe"),
        _record(product_type="tile-coadd"),
    ]
    assert len({record.spectrum_id for record in records}) == len(records)
    assert len({record.filename for record in records}) == len(records)


def test_near_duplicate_overlap_is_detected_relative_to_grid_spacing() -> None:
    wavelength = np.array([5000.0, 5000.8, 5001.6, 5001.6000000004, 5002.4])
    assert count_tolerance_duplicates(wavelength) == 1
    assert count_tolerance_duplicates(np.array([5000.0, 5000.8, 5001.6, 5002.4])) == 0


def _provenance(targetid: int) -> np.ndarray:
    dtype = [("TARGETID", "i8"), ("NIGHT", "i8"), ("EXPID", "i8"), ("TILEID", "i8"), ("MJD", "f8")]
    return np.array(
        [
            (targetid, 20220127, 120463, 21686, 59607.54638998),
            (targetid, 20220304, 124922, 23252, 59643.44186746),
        ],
        dtype=dtype,
    )


def test_exposure_summary_preserves_multi_exposure_multi_night_semantics() -> None:
    provenance = _provenance(39627842660599483)
    summary = exposure_summary(provenance)

    assert summary["n_exp"] == 2
    assert summary["n_night"] == 2
    assert summary["mjd_min"] == pytest.approx(59607.54638998)
    assert summary["mjd_max"] == pytest.approx(59643.44186746)
    assert summary["mjd_effective"] == pytest.approx(np.mean(provenance["MJD"]))


def _camera_fixture() -> tuple[dict[str, np.ndarray], ...]:
    wave = {
        "b": np.arange(3600.0, 5801.0, 0.8),
        "r": np.arange(5760.0, 7621.0, 0.8),
        "z": np.arange(7520.0, 9824.1, 0.8),
    }
    flux = {band: (np.sin(values / 700.0) + 3.0)[None, :] for band, values in wave.items()}
    ivar = {band: np.full((1, values.size), index + 1.0) for index, (band, values) in enumerate(wave.items())}
    mask = {band: np.zeros((1, values.size), dtype=np.int32) for band, values in wave.items()}
    resolution = {}
    for band, values in wave.items():
        matrix = np.zeros((1, 3, values.size))
        matrix[:, 1, :] = 1.0
        resolution[band] = matrix
    return wave, flux, ivar, mask, resolution


def test_wrapper_matches_direct_official_camera_reference() -> None:
    from desispec.coaddition import coadd_cameras
    from desispec.spectra import Spectra

    targetid = 39627842660599483
    wave, flux, ivar, mask, resolution = _camera_fixture()
    fibermap = np.array([(targetid,)], dtype=[("TARGETID", "i8")])
    provenance = _provenance(targetid)

    expected = coadd_cameras(
        Spectra(
            bands=["b", "r", "z"],
            wave=wave,
            flux=flux,
            ivar=ivar,
            mask=mask,
            resolution_data=resolution,
            fibermap=fibermap,
            exp_fibermap=provenance,
        )
    )
    actual = merge_cameras_official(
        wave=wave,
        flux=flux,
        ivar=ivar,
        mask=mask,
        resolution_data=resolution,
        fibermap=fibermap,
        exp_fibermap=provenance,
    )
    band = expected.bands[0]

    assert np.array_equal(actual["wavelength"], expected.wave[band])
    assert np.array_equal(actual["flux"], expected.flux[band][0])
    assert np.array_equal(actual["ivar"], expected.ivar[band][0])
    assert np.array_equal(actual["mask"], expected.mask[band][0])
    assert np.array_equal(actual["resolution"], expected.resolution_data[band][0])
    assert np.array_equal(actual["exp_fibermap"], provenance)
