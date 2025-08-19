"""
extract resampled images:
plenopticam -f <path_to_lfr_file> -c <path_to_calibration_tar>

"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

import plenopticam as pcam
from plenopticam.cfg import PlenopticamConfig
from plenopticam.lfp_aligner import LfpAligner
from plenopticam.lfp_calibrator import CaliFinder, LfpCalibrator
from plenopticam.lfp_extractor import LfpExtractor


@dataclass
class SAIInfo:
    u: int
    v: int
    path: Path


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("$", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    if proc.stdout:
        print(proc.stdout)


def ensure_empty(dirpath: Path):
    if dirpath.exists():
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)


def decode_lfr(lfr_path: Path, calib_tar: Path) -> np.ndarray:
    cfg = PlenopticamConfig()
    cfg.default_values()
    cfg.params[cfg.lfp_path] = str(lfr_path.resolve())
    cfg.params[cfg.cal_path] = str(calib_tar.resolve())
    cfg.params[cfg.opt_cali] = True
    cfg.params[cfg.ptc_leng] = 13
    cfg.params[cfg.cal_meth] = pcam.cfg.constants.CALI_METH[3]

    sta = pcam.misc.PlenopticamStatus()
    ensure_empty(cfg.exp_path)

    reader = pcam.lfp_reader.LfpReader(cfg, sta)
    reader.main()
    finder = CaliFinder(cfg, sta)
    finder.main()
    meta_cond = not (
        os.path.exists(cfg.params[cfg.cal_meta])
        and cfg.params[cfg.cal_meta].lower().endswith("json")
    )
    if meta_cond or cfg.params[cfg.opt_cali]:
        calibrator = LfpCalibrator(finder.wht_bay, cfg, sta)
        calibrator.main()
        cfg = calibrator.cfg

    cfg.load_cal_data()
    if cfg.cond_lfp_align():
        aligner = LfpAligner(reader.lfp_img, cfg, sta, finder.wht_bay)
        aligner.main()

    extractor = LfpExtractor(aligner.lfp_img, cfg, sta)
    extractor.main()
    return extractor.vp_img_arr


def main():
    parser = argparse.ArgumentParser(
        description="Lytro .lfr -> SAIs -> COLMAP rig reconstruction"
    )
    parser.add_argument(
        "--lfr", type=Path, required=True, help="Path to Lytro Illum .lfr"
    )
    parser.add_argument(
        "--calib_tar",
        type=Path,
        default=None,
        help="Path to Lytro caldata-XXX.tar (or folder)",
    )
    args = parser.parse_args()

    images = decode_lfr(args.lfr, args.calib_tar)  # 13x13 grid of images  # noqa: F841
    # use only inner 11x11 grid, add argument that toggles this


if __name__ == "__main__":
    main()
