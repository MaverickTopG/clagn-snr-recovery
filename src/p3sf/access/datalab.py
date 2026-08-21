# =====================================================================
# VENDORED CODE — do not sync with upstream; modify freely here.
#   source repo : CLAGN-WISE-ApJ (author's earlier project)
#   source file : src/clagn_apj/datalab.py
#   upstream rev: c3bd1d4ece9f64ccf7ab592880ed7da7832e6f31
#   copied      : 2026-08-14
#   sha256(src) : 3c88e585a44036bbab1caf19bad9728f4bcb02f544aa8bc5b1bacd550908d972
# =====================================================================
from __future__ import annotations

import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from p3sf.access.cohort_config import CohortConfig
from p3sf.infra.provenance import canonical_hash, sha256_file


class DataLabError(RuntimeError):
    """A remote Data Lab operation failed."""


class DataLabAuthenticationError(DataLabError):
    """No authenticated Data Lab session is available."""


class DataLabGateway(Protocol):
    def username(self) -> str: ...

    def tables(self) -> set[str]: ...

    def import_sources(self, table: str, csv_path: Path) -> None: ...

    def index_sources(self, table: str) -> None: ...

    def row_count(self, table: str) -> int: ...

    def submit(self, query: str, output_table: str) -> str: ...

    def status(self, job_id: str) -> str: ...

    def error(self, job_id: str) -> str: ...

    def index_column(self, table: str, column: str) -> None: ...

    def query_csv(self, query: str) -> str: ...


class AstroDataLabGateway:
    """Thin authenticated adapter around astro-datalab."""

    def __init__(self, profile: str = "default"):
        from dl import authClient, queryClient  # type: ignore[import-untyped]

        username = str(authClient.whoAmI())
        if username == "anonymous":
            raise DataLabAuthenticationError(
                "NOIRLab Data Lab is not authenticated; run `datalab login` locally"
            )
        self._auth = authClient
        self._query = queryClient
        self._token = str(authClient.def_token(None))
        if not bool(authClient.isValidToken(self._token)):
            raise DataLabAuthenticationError("the saved NOIRLab Data Lab token is invalid")
        self._username = username
        self._profile = profile

    def _check(self, response: object, operation: str) -> str:
        text = str(response)
        lowered = text.lower()
        if "<html" in lowered or "error" in lowered or "exception" in lowered:
            raise DataLabError(f"{operation} failed: {text[:1000]}")
        return text

    def username(self) -> str:
        return self._username

    def tables(self) -> set[str]:
        listing = self._check(
            self._query.mydb_list(self._token, table=""), "list MyDB tables"
        )
        return {
            item.strip().split(",", maxsplit=1)[0]
            for item in listing.replace("\n", ",").split(",")
            if item.strip() and not item.strip().startswith("created:")
        }

    def import_sources(self, table: str, csv_path: Path) -> None:
        response = self._query.mydb_import(
            self._token,
            table,
            str(csv_path),
            schema=(
                "source_id,text\n"
                "ra,double precision\n"
                "dec,double precision\n"
                "redshift,double precision"
            ),
        )
        self._check(response, f"import {table}")

    def index_sources(self, table: str) -> None:
        self._check(
            self._query.mydb_index(self._token, table, "source_id"),
            f"index {table}.source_id",
        )
        self._check(
            self._query.mydb_index(
                self._token,
                table,
                "",
                q3c="ra,dec",
                cluster=True,
                async_=False,
            ),
            f"Q3C index {table}",
        )

    def row_count(self, table: str) -> int:
        result = self.query_csv(f"SELECT COUNT(*) AS n FROM mydb://{table}")
        return int(pd.read_csv(io.StringIO(result)).iloc[0]["n"])

    def submit(self, query: str, output_table: str) -> str:
        response = self._query.query(
            self._token,
            sql=query,
            fmt="csv",
            out=f"mydb://{output_table}",
            async_=True,
            drop=False,
            profile=self._profile,
        )
        return self._check(response, "submit asynchronous cohort query").strip()

    def status(self, job_id: str) -> str:
        return self._check(
            self._query.status(self._token, jobId=job_id), f"status for {job_id}"
        ).strip()

    def error(self, job_id: str) -> str:
        return str(self._query.error(self._token, jobId=job_id))

    def index_column(self, table: str, column: str) -> None:
        self._check(
            self._query.mydb_index(self._token, table, column),
            f"index {table}.{column}",
        )

    def query_csv(self, query: str) -> str:
        response = self._query.query(
            self._token,
            sql=query,
            fmt="csv",
            profile=self._profile,
        )
        return self._check(response, "synchronous MyDB query")


@dataclass(frozen=True)
class AcquisitionState:
    source_sha256: str
    source_rows: int
    source_table: str
    result_table: str
    query_sha256: str
    job_id: str
    username: str
    status: str


def build_cohort_query(source_table: str, config: CohortConfig) -> str:
    radius_degrees = config.parent_match_radius_arcsec / 3600.0
    return f"""
SELECT
  p.source_id,
  p.ra AS source_ra,
  p.dec AS source_dec,
  p.redshift AS parent_redshift,
  CAST(s.specobjid AS VARCHAR) AS sdss_specobjid,
  s.ra AS sdss_ra,
  s.dec AS sdss_dec,
  s.mjd AS sdss_mjd,
  s.plate AS sdss_plate,
  s.fiberid AS sdss_fiberid,
  s.run2d AS sdss_run2d,
  s.survey AS sdss_survey,
  s.instrument AS sdss_instrument,
  s.wavemin AS sdss_wavemin,
  s.wavemax AS sdss_wavemax,
  s.zwarning AS sdss_zwarning,
  s.scienceprimary AS sdss_scienceprimary,
  s.snmedian AS sdss_snmedian,
  CAST(d.id AS VARCHAR) AS desi_zpix_id,
  CAST(d.targetid AS VARCHAR) AS desi_targetid,
  d.survey AS desi_survey,
  d.program AS desi_program,
  d.healpix AS desi_healpix,
  d.min_mjd AS desi_min_mjd,
  d.mean_mjd AS desi_mean_mjd,
  d.max_mjd AS desi_max_mjd,
  d.z AS desi_z,
  d.zwarn AS desi_zwarn,
  d.zcat_primary AS desi_zcat_primary,
  d.coadd_fiberstatus AS desi_coadd_fiberstatus,
  d.coadd_numexp AS desi_coadd_numexp,
  d.coadd_numnight AS desi_coadd_numnight,
  d.tsnr2_qso AS desi_tsnr2_qso,
  d.mean_psf_to_fiber_specflux AS desi_mean_psf_to_fiber_specflux,
  q3c_dist(p.ra, p.dec, s.ra, s.dec) * 3600.0 AS parent_sdss_sep_arcsec,
  x.distance AS sdss_desi_sep_arcsec
FROM mydb://{source_table} AS p
JOIN {config.sdss_table} AS s
  ON q3c_join(p.ra, p.dec, s.ra, s.dec, {radius_degrees:.12f}) = 'true'
JOIN {config.sdss_desi_xmatch_table} AS x
  ON s.specobjid = x.id1
JOIN {config.desi_table} AS d
  ON x.id2 = d.id
""".strip()


def state_identity(source_path: Path, source_rows: int, query: str) -> dict[str, str]:
    source_hash = sha256_file(source_path)
    query_hash = canonical_hash({"query": query})
    return {
        "source_sha256": source_hash,
        "source_rows": str(source_rows),
        "source_table": f"clagn_apj_sources_{source_hash[:12]}",
        "result_table": f"clagn_apj_pairs_{query_hash[:12]}",
        "query_sha256": query_hash,
    }


def write_state(path: Path, state: AcquisitionState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_state(path: Path) -> AcquisitionState:
    return AcquisitionState(**json.loads(path.read_text()))


def wait_for_job(
    gateway: DataLabGateway,
    state_path: Path,
    state: AcquisitionState,
    *,
    poll_seconds: float = 10.0,
) -> AcquisitionState:
    current = state
    while current.status not in {"COMPLETED", "ERROR", "ABORTED"}:
        status = gateway.status(current.job_id).upper()
        current = AcquisitionState(**{**asdict(current), "status": status})
        write_state(state_path, current)
        if status not in {"COMPLETED", "ERROR", "ABORTED"}:
            time.sleep(poll_seconds)
    if current.status != "COMPLETED":
        raise DataLabError(
            f"cohort query {current.job_id} ended as {current.status}: "
            f"{gateway.error(current.job_id)[:2000]}"
        )
    return current


def acquire_remote_matches(
    gateway: DataLabGateway,
    *,
    sources: pd.DataFrame,
    source_path: Path,
    query: str,
    interim_dir: Path,
    chunk_size: int,
    resume: bool,
    poll_seconds: float = 10.0,
) -> tuple[list[Path], AcquisitionState]:
    identity = state_identity(source_path, len(sources), query)
    state_path = interim_dir / "query_job.json"
    source_csv = interim_dir / "sources_upload.csv"
    chunks_dir = interim_dir / "chunks"
    interim_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if state_path.exists():
        if not resume:
            raise FileExistsError(
                f"resumable state exists at {state_path}; rerun with --resume"
            )
        state = read_state(state_path)
        expected = {
            "source_sha256": identity["source_sha256"],
            "source_rows": int(identity["source_rows"]),
            "source_table": identity["source_table"],
            "result_table": identity["result_table"],
            "query_sha256": identity["query_sha256"],
        }
        for key, value in expected.items():
            if getattr(state, key) != value:
                raise DataLabError(f"resume state mismatch for {key}")
    else:
        available = gateway.tables()
        source_table = identity["source_table"]
        if source_table not in available:
            sources.to_csv(source_csv, index=False)
            gateway.import_sources(source_table, source_csv)
            gateway.index_sources(source_table)
        if gateway.row_count(source_table) != len(sources):
            raise DataLabError(
                f"MyDB source table {source_table} does not contain {len(sources)} rows"
            )
        result_table = identity["result_table"]
        if result_table in available:
            state = AcquisitionState(
                source_sha256=identity["source_sha256"],
                source_rows=len(sources),
                source_table=source_table,
                result_table=result_table,
                query_sha256=identity["query_sha256"],
                job_id="reused-existing-result",
                username=gateway.username(),
                status="COMPLETED",
            )
        else:
            job_id = gateway.submit(query, result_table)
            state = AcquisitionState(
                source_sha256=identity["source_sha256"],
                source_rows=len(sources),
                source_table=source_table,
                result_table=result_table,
                query_sha256=identity["query_sha256"],
                job_id=job_id,
                username=gateway.username(),
                status="SUBMITTED",
            )
        write_state(state_path, state)

    if state.status != "COMPLETED":
        state = wait_for_job(gateway, state_path, state, poll_seconds=poll_seconds)
    gateway.index_column(state.result_table, "source_id")

    identifiers = sources["source_id"].astype(str).sort_values(kind="mergesort").tolist()
    chunk_paths: list[Path] = []
    for start in range(0, len(identifiers), chunk_size):
        stop = min(start + chunk_size, len(identifiers))
        lower = identifiers[start]
        upper = identifiers[stop] if stop < len(identifiers) else None
        if "'" in lower or (upper is not None and "'" in upper):
            raise ValueError("source identifiers containing quotes are unsupported")
        chunk_path = chunks_dir / f"matches_{start:07d}_{stop:07d}.csv"
        chunk_paths.append(chunk_path)
        if chunk_path.exists() and resume:
            continue
        condition = f"source_id >= '{lower}'"
        if upper is not None:
            condition += f" AND source_id < '{upper}'"
        csv_text = gateway.query_csv(
            f"SELECT * FROM mydb://{state.result_table} "
            f"WHERE {condition} ORDER BY source_id"
        )
        temporary = chunk_path.with_suffix(".tmp")
        temporary.write_text(csv_text)
        temporary.replace(chunk_path)
    return chunk_paths, state
