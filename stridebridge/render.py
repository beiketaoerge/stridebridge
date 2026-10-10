"""Render the recorded continuous physics trajectory with phase/metric overlays."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def font(size):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def phase_label(t, weight, events):
    entry = next((e["time"] for e in events if e["event"] == "request:motion"), float("inf"))
    leave = next((e["time"] for e in events if e["event"] == "request:locomotion"), float("inf"))
    if 1e-6 < weight < 1-1e-6:
        return "BLEND TO MOTION" if t < leave else "BLEND TO LOCOMOTION"
    if weight >= 1-1e-6:
        return "MOTION IMITATION"
    if entry <= t < leave:
        return "ACTIVE BALANCE / ENTRY REQUESTED"
    return "LOCOMOTION / BALANCE" if t < entry else "RETURN TO LOCOMOTION"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    manifest = json.loads((args.run/"manifest.json").read_text())
    metrics = json.loads((args.run/"metrics.json").read_text())
    with np.load(args.run/"rollout.npz", allow_pickle=False) as recording:
        log = {k: recording[k] for k in recording.files}
    if not len(log["time"]):
        raise ValueError("Cannot render an empty recording")
    model = mujoco.MjModel.from_xml_path(str(ROOT/manifest["scene"]))
    robot = manifest["args"]["robot"]
    if robot == "h1":
        board = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "coretest_stairhome_obstacle_0p30")
        if board >= 0:
            model.geom_rgba[board, 3] = 0
            model.geom_contype[board] = model.geom_conaffinity[board] = 0
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.8 if robot == "h1" else 2.7
    camera.azimuth = 145 if robot == "h1" else 125
    camera.elevation = -15
    camera.lookat[:] = [0, 0, .85 if robot == "h1" else .65]
    output = args.output or args.run/"demo.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    title_font, body_font, small_font = font(24), font(18), font(14)
    duration = float(log["time"][-1] + .02)
    first = None
    try:
        with imageio.get_writer(output, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=16, ffmpeg_params=["-movflags", "+faststart"]) as writer:
            for t in np.arange(0, duration-1e-8, 1/args.fps):
                i = min(int(np.searchsorted(log["time"], t, side="right"))-1, len(log["time"])-1)
                i = max(0, i)
                data.qpos[:] = log["qpos"][i]
                data.qvel[:] = log["qvel"][i]
                mujoco.mj_forward(model, data)
                camera.lookat[:2] = data.qpos[:2]
                renderer.update_scene(data, camera=camera)
                img = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(img, "RGBA")
                draw.rectangle([0, 0, args.width, 98], fill=(13, 21, 35, 238))
                draw.text((24, 12), "STRIDEBRIDGE  /  " + robot.upper(), font=title_font, fill=(238, 244, 250))
                phase = phase_label(t, log["motion_weight"][i], metrics["events"])
                draw.text((24, 50), phase, font=body_font, fill=(78, 211, 185))
                draw.text((args.width-240, 16), f"{t:05.2f} s  /  {duration:.1f} s", font=body_font, fill="white")
                transport = "MuJoCo + DDS  |  loopback" if manifest.get("transport") == "dds" else "MuJoCo  |  continuous rollout"
                draw.text((args.width-240, 50), transport, font=small_font, fill=(173, 190, 207))
                y = args.height-82
                draw.rectangle([0, y, args.width, args.height], fill=(13, 21, 35, 240))
                speed = float(log["velocity"][i, 0])
                draw.text((24, y+13), f"Tilt {log['tilt'][i]:4.1f}°    Base {log['height'][i]:.2f} m    Vx {speed:+.2f} m/s    Cmd {log['command'][i, 0]:.2f} m/s", font=body_font, fill="white")
                footer = "Flat ground · active balance · no support forces"
                if manifest.get("transport") == "dds":
                    footer = (f"DDS / lo · 500 Hz physics · 50 Hz policy · "
                              f"Feedback age {log['feedback_age'][i]*1000:.1f} ms · "
                              f"Command age {log['command_age'][i]*1000:.1f} ms")
                draw.text((24, y+45), footer, font=small_font, fill=(173, 190, 207))
                x0, x1 = 24, args.width-24
                draw.rectangle([x0, args.height-12, x1, args.height-7], fill=(54, 66, 84))
                draw.rectangle([x0, args.height-12, x0+(x1-x0)*t/duration, args.height-7], fill=(78, 211, 185))
                frame = np.asarray(img)
                writer.append_data(frame)
                if first is None:
                    first = img.copy()
        if first:
            first.save(output.with_suffix(".jpg"), quality=90)
    finally:
        renderer.close()
    print(output)


if __name__ == "__main__":
    main()
