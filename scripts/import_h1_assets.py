#!/usr/bin/env python3
"""Import the user's existing H1 checkpoints into ignored, portable asset paths."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml

ROOT = Path(__file__).resolve().parents[1]
CROSS = "h1_motion_104_coretestvideo_gmr_stairhome_obstacle0p30_yawm76p5_model30000.yaml"
WALK = "h1_dwaq_0816_box3kg.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-bundle", required=True, type=Path, help="Existing 0927 bundle root")
    parser.add_argument("--locomotion-bundle", required=True, type=Path, help="Existing h1_deploy_0823 root")
    args = parser.parse_args()
    destination = ROOT / "assets/local/h1"
    records = []
    for root, label, config in [(args.motion_bundle.resolve(), "0927", CROSS),
                                (args.locomotion_bundle.resolve(), "0823", WALK)]:
        cfg_path = root / "deploy/deploy_real/configs" / config
        cfg = yaml.safe_load(cfg_path.read_text())
        paths = [cfg_path]
        for key in ["policy_path", "motion_file"]:
            if key in cfg:
                paths.append(Path(cfg[key].replace("{LEGGED_GYM_ROOT_DIR}", str(root))))
        if label == "0927":
            paths += [root/"deploy/deploy_real/deploy_real_h1_motion_104_realfix.py"]
            paths += sorted((root/"deploy/deploy_real/common").glob("*.py"))
            paths += [p for p in sorted((root/"unitree_mujoco/unitree_robots/h1").rglob("*"))
                      if p.is_file() and p.suffix.lower() in {".xml", ".stl", ".obj", ".png"}]
        for source in paths:
            relative = source.resolve().relative_to(root)
            target = destination / label / relative
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise SystemExit("Existing asset differs; refusing to overwrite: " + str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records.append({"path": target.relative_to(destination).as_posix(), "sha256": digest,
                            "bytes": target.stat().st_size})
    (destination/"manifest.json").write_text(json.dumps(records, indent=2)+"\n")
    print("Imported", len(records), "files into", destination)
    print("Local weights and motion data are excluded from Git. See docs/models.md.")


if __name__ == "__main__":
    main()
