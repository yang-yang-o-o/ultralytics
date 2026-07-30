#!/usr/bin/env python3
"""Evaluate IC-BIN BOP CSV with bop_toolkit (BOP19 localization metrics).

Sets BOP_PATH / BOP_RESULTS_PATH / BOP_EVAL_PATH and runs scripts/eval_bop19_pose.py.

Example:
  python examples/icbin_seg6d_eval_bop.py \\
    --result-csv runs/segment/icbin-seg/icbin-seg6d-smoke/bop_results/seg6d-smoke_icbin-test.csv

Optional GT-oracle smoke (perfect poses → AR≈1):
  python examples/icbin_seg6d_eval_bop.py --export-gt-oracle --skip-model-export
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/ultralytics")
BOP_TOOLKIT = Path("/root/bop_toolkit")
BOP_DATASETS = Path("/root/YOLO6D/datasets/bop")
ICBIN = BOP_DATASETS / "icbin"
DEFAULT_RESULTS = ROOT / "runs" / "segment" / "icbin-seg" / "icbin-seg6d-smoke" / "bop_results"
DEFAULT_EVAL = ROOT / "runs" / "segment" / "icbin-seg" / "icbin-seg6d-smoke" / "bop_eval"


def export_gt_oracle(out_csv: Path, targets_path: Path, split: str = "test") -> Path:
    """Write GT poses for BOP19 targets (sanity-check the eval pipeline)."""
    targets = json.loads(targets_path.read_text())
    rows = ["scene_id,im_id,obj_id,score,R,t,time"]
    # cache scene GT
    scene_gt: dict[int, dict] = {}
    for t in targets:
        scene_id = int(t["scene_id"])
        im_id = int(t["im_id"])
        obj_id = int(t["obj_id"])
        inst_count = int(t.get("inst_count", 1))
        if scene_id not in scene_gt:
            scene_gt[scene_id] = json.loads(
                (ICBIN / split / f"{scene_id:06d}" / "scene_gt.json").read_text()
            )
        gts = scene_gt[scene_id][str(im_id)]
        # take first matching obj_id instances up to inst_count
        matched = 0
        for gt in gts:
            if int(gt["obj_id"]) != obj_id:
                continue
            R = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            tt = np.array(gt["cam_t_m2c"], dtype=np.float64).reshape(3)
            R_str = " ".join(f"{v:.8f}" for v in R.reshape(-1))
            t_str = " ".join(f"{v:.8f}" for v in tt.reshape(-1))
            rows.append(f"{scene_id},{im_id},{obj_id},1.000000,{R_str},{t_str},0.001000")
            matched += 1
            if matched >= inst_count:
                break
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(rows) + "\n")
    print(f"GT oracle wrote {out_csv} ({len(rows)-1} poses)")
    return out_csv


def ensure_targets_layout() -> None:
    """bop_toolkit expects <BOP_PATH>/icbin/test_targets_bop19.json."""
    src = ICBIN / "test_targets_bop19.json"
    if not src.exists():
        alt = ICBIN / "icbin" / "test_targets_bop19.json"
        if alt.exists():
            src = alt
            (ICBIN / "test_targets_bop19.json").write_text(alt.read_text())
    if not (ICBIN / "test_targets_bop19.json").exists():
        raise FileNotFoundError("test_targets_bop19.json missing under icbin/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--result-csv",
        type=Path,
        default=DEFAULT_RESULTS / "seg6d-smoke_icbin-test.csv",
    )
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_RESULTS.parent / "weights" / "best.pt")
    parser.add_argument("--export-gt-oracle", action="store_true")
    parser.add_argument("--skip-model-export", action="store_true", help="Do not run model→CSV export")
    parser.add_argument("--renderer-type", default="vispy")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full BOP19 (VSD+MSSD+MSPD). Default: lite MSSD+MSPD only (faster smoke).",
    )
    args = parser.parse_args()

    ensure_targets_layout()
    args.results_path.mkdir(parents=True, exist_ok=True)
    args.eval_path.mkdir(parents=True, exist_ok=True)

    result_csv = args.result_csv
    if args.export_gt_oracle:
        result_csv = args.results_path / "gt-oracle_icbin-test.csv"
        export_gt_oracle(result_csv, ICBIN / "test_targets_bop19.json")
    elif not args.skip_model_export:
        export_py = ROOT / "examples" / "icbin_seg6d_export_bop.py"
        cmd = [
            sys.executable,
            str(export_py),
            "--weights",
            str(args.weights),
            "--out",
            str(result_csv),
            "--conf",
            str(args.conf),
        ]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)

    if not result_csv.exists():
        raise SystemExit(f"result csv missing: {result_csv}")

    env = os.environ.copy()
    env["BOP_PATH"] = str(BOP_DATASETS)
    env["BOP_RESULTS_PATH"] = str(args.results_path)
    env["BOP_EVAL_PATH"] = str(args.eval_path)
    env["BOP_NUM_WORKERS"] = str(args.num_workers)
    # eval_bop19_pose spawns `python scripts/eval_calc_errors.py` — force venv
    venv_bin = str(Path(sys.executable).resolve().parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(BOP_TOOLKIT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    # Prefer relative filename under results_path
    result_name = result_csv.name
    if result_csv.parent.resolve() != args.results_path.resolve():
        # copy/symlink into results_path
        dest = args.results_path / result_name
        dest.write_bytes(result_csv.read_bytes())
        result_name = dest.name

    eval_script = (
        BOP_TOOLKIT / "scripts" / "eval_bop19_pose.py"
        if args.full
        else ROOT / "examples" / "eval_bop19_pose_lite.py"
    )
    cmd = [
        sys.executable,
        str(eval_script),
        "--result_filenames",
        result_name,
        "--results_path",
        str(args.results_path),
        "--eval_path",
        str(args.eval_path),
        "--renderer_type",
        args.renderer_type,
        "--num_workers",
        str(args.num_workers),
        "--targets_filename",
        "test_targets_bop19.json",
    ]
    print("Running:", " ".join(cmd))
    print(f"BOP_PATH={env['BOP_PATH']}")
    rc = subprocess.call(cmd, env=env, cwd=str(BOP_TOOLKIT))
    if rc != 0:
        raise SystemExit(rc)

    # print scores if present
    score_files = sorted(args.eval_path.rglob("scores_*.json"))
    for sf in score_files[-6:]:
        print(f"--- {sf.relative_to(args.eval_path)} ---")
        try:
            print(sf.read_text()[:500])
        except Exception:
            pass
    print(f"eval output: {args.eval_path}")


if __name__ == "__main__":
    main()
