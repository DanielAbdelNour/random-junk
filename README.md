
"""Dataset generation and verification utilities."""

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

from utils.simulation import generate_blob_noise, world_to_yolo_polar_aware, SimTarget
from utils.viz import build_trails_composite


def _compose_trails_from_frames(
    frames_u8,
    frame_idx,
    n_previous,
    alpha_decay,
    black_threshold,
):
    start_frame = max(0, frame_idx - n_previous)
    frames_to_show = range(start_frame, frame_idx + 1)

    num_frames = len(frames_to_show)
    opacities = [alpha_decay ** (num_frames - 1 - i) for i in range(num_frames)]

    h_px, w_px = frames_u8.shape[1:3]
    composite = np.zeros((h_px, w_px, 3), dtype=np.float32)
    n_imgs = len(frames_to_show)

    # Previous frames
    for k, (fid, opacity) in enumerate(zip(frames_to_show, opacities)):
        if k >= n_imgs - 1:
            continue
        img = frames_u8[fid].astype(np.float32) / 255.0
        mask = np.all(img * 255.0 <= black_threshold, axis=2)
        img[mask] = 0.0
        composite += img * opacity

    # Current frame (red)
    current = frames_u8[frame_idx].astype(np.float32) / 255.0
    mask = np.all(current * 255.0 <= black_threshold, axis=2)
    current[mask] = 0.0
    alpha = current.mean(axis=2)
    alpha = np.clip(alpha * opacities[-1], 0, 1)

    composite[:, :, 0] = composite[:, :, 0] * (1 - alpha) + current[:, :, 0] * alpha
    composite[:, :, 1] = composite[:, :, 1] * (1 - alpha)
    composite[:, :, 2] = composite[:, :, 2] * (1 - alpha)

    composite = np.clip(composite, 0, 1)
    return (composite * 255.0).astype(np.uint8)


def _worker_build_trail(args):
    (
        shm_name,
        shape,
        dtype,
        frame_idx,
        n_previous,
        alpha_decay,
        black_threshold,
    ) = args

    shm = shared_memory.SharedMemory(name=shm_name)
    frames_u8 = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    out = _compose_trails_from_frames(
        frames_u8=frames_u8,
        frame_idx=frame_idx,
        n_previous=n_previous,
        alpha_decay=alpha_decay,
        black_threshold=black_threshold,
    )
    shm.close()
    return frame_idx, out
from .math import db2pow, pow2db, gaussian_beam


@dataclass
class DatasetConfig:
    # Radar Parameters
    max_range: float = 2500.0
    min_range: float = 100.0
    range_res: float = 5.0
    beamwidth: float = 1.0
    scan_period: float = 1.0

    # Simulation parameters
    num_pulses: int = 360
    noise_floor_db: float = -110.0
    num_frames: int = 5000
    output_dir: str = "radar_dataset2"

    # Image output settings
    img_size_px: int = 640
    dpi: int = 100

    # Dynamic vessel population
    dynamic_vessels: bool = True
    initial_vessels: int = 3
    max_vessels: int = 6
    spawn_probability_per_frame: float = 0.015
    spawn_at_range_factor: float = 1.04
    remove_beyond_range_factor: float = 1.5

    # Blob Noise Parameters
    blob_noise_enabled: bool = True
    blob_noise_probability: float = 0.5
    blob_noise_count_range: tuple = (1, 3)
    blob_noise_size_range: tuple = (1.0, 10.0)
    blob_noise_rcs_db_range: tuple = (25.0, 40.0)

    # Output handling
    overwrite: bool = True


def _prepare_output_dirs(output_dir, overwrite=True):
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")

    for d in [output_dir, img_dir, lbl_dir]:
        if os.path.exists(d):
            if overwrite:
                shutil.rmtree(d)
            else:
                raise FileExistsError(f"Directory exists: {d}")
        os.makedirs(d)

    return img_dir, lbl_dir


def generate_dataset(config: DatasetConfig):
    """Generate radar dataset images and YOLO labels based on config."""

    img_dir, lbl_dir = _prepare_output_dirs(config.output_dir, config.overwrite)

    range_bins = np.arange(config.min_range, config.max_range, config.range_res)
    num_range_bins = len(range_bins)
    az_angles = np.linspace(0, 360, config.num_pulses, endpoint=False)
    range_loss = 1.0 / (range_bins**4)

    # Initial vessel population and ID counter for dynamic spawning
    targets = [
        SimTarget(i, range_res=config.range_res, max_range=config.max_range)
        for i in range(config.initial_vessels)
    ]
    next_vessel_id = config.initial_vessels

    fig_size = config.img_size_px / config.dpi
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=config.dpi, facecolor="black")

    ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
    ax.axis("off")

    clutter_base = 1.0 / (range_bins**3)
    clutter_base = clutter_base / clutter_base[0] * db2pow(config.noise_floor_db + 25)

    az_rads = np.deg2rad(az_angles)
    r_grid, az_grid = np.meshgrid(range_bins, az_rads)
    x_grid = r_grid * np.cos(az_grid)
    y_grid = r_grid * np.sin(az_grid)

    mesh = ax.pcolormesh(
        x_grid,
        y_grid,
        np.zeros((config.num_pulses, num_range_bins)),
        cmap="gray",
        shading="auto",
        vmin=-105,
        vmax=-50,
    )

    ax.set_aspect("equal")
    ax.set_xlim(-config.max_range, config.max_range)
    ax.set_ylim(-config.max_range, config.max_range)

    for i in tqdm(range(config.num_frames), desc="Generating frames", unit="frame"):
        # Dynamic vessels: maybe spawn a new one entering the radar area
        if config.dynamic_vessels and len(targets) < config.max_vessels:
            if np.random.random() < config.spawn_probability_per_frame:
                spawn_range = config.max_range * config.spawn_at_range_factor
                new_tgt = SimTarget(
                    next_vessel_id,
                    spawn_range=spawn_range,
                    range_res=config.range_res,
                    max_range=config.max_range,
                )
                targets.append(new_tgt)
                next_vessel_id += 1

        noise = np.random.exponential(
            db2pow(config.noise_floor_db), (config.num_pulses, num_range_bins)
        )
        clutter = np.random.exponential(clutter_base, (config.num_pulses, num_range_bins))
        scan_data = noise + clutter

        if config.blob_noise_enabled:
            generate_blob_noise(
                scan_data=scan_data,
                az_angles=az_angles,
                range_bins=range_bins,
                range_loss=range_loss,
                beamwidth=config.beamwidth,
                min_range=config.min_range,
                range_res=config.range_res,
                probability=config.blob_noise_probability,
                count_range=config.blob_noise_count_range,
                size_range=config.blob_noise_size_range,
                rcs_db_range=config.blob_noise_rcs_db_range,
            )

        yolo_lines = []

        for pulse_idx, az_deg in enumerate(az_angles):
            dt_pulse = (pulse_idx / config.num_pulses) * config.scan_period
            for tgt in targets:
                pos_now = tgt.pos + tgt.vel * dt_pulse

                extent_points, extent_weights = tgt.get_extent_points(pos_now)

                for pt, weight in zip(extent_points, extent_weights):
                    rng = np.linalg.norm(pt)

                    if rng < config.min_range or rng > config.max_range:
                        continue

                    pt_az = np.rad2deg(np.arctan2(pt[1], pt[0])) % 360
                    angle_diff = (az_deg - pt_az + 180) % 360 - 180

                    if abs(angle_diff) < config.beamwidth * 1.5:
                        gain = gaussian_beam(angle_diff, config.beamwidth)
                        bin_idx_center = (rng - config.min_range) / config.range_res

                        for bin_offset in [-1, 0, 1]:
                            bin_idx = int(bin_idx_center) + bin_offset
                            if 0 <= bin_idx < num_range_bins:
                                bin_range = config.min_range + bin_idx * config.range_res
                                range_diff = abs(rng - bin_range)
                                range_weight = np.exp(
                                    -(range_diff**2) / (2 * (config.range_res * 0.7) ** 2)
                                )

                                p_tgt = (
                                    db2pow(tgt.rcs_db)
                                    * weight
                                    * range_weight
                                    * range_loss[bin_idx]
                                    * (gain**2)
                                    * 1e14
                                )
                                scan_data[pulse_idx, bin_idx] += p_tgt

        for tgt in targets:
            rng = np.linalg.norm(tgt.pos)
            if config.min_range < rng < config.max_range:
                nx, ny, nw, nh = world_to_yolo_polar_aware(
                    tgt.pos,
                    tgt.vel,
                    config.scan_period,
                    tgt.length,
                    tgt.width,
                    max_range=config.max_range,
                    beamwidth_deg=config.beamwidth,
                )

                nx = np.clip(nx, 0, 1)
                ny = np.clip(ny, 0, 1)
                nw = np.clip(nw, 0, 1)
                nh = np.clip(nh, 0, 1)

                yolo_lines.append(f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

        scan_db = pow2db(scan_data)
        mesh.set_array(scan_db.ravel())

        img_filename = os.path.join(img_dir, f"frame_{i:04d}.png")
        plt.savefig(img_filename, dpi=config.dpi)

        lbl_filename = os.path.join(lbl_dir, f"frame_{i:04d}.txt")
        with open(lbl_filename, "w") as f:
            f.write("\n".join(yolo_lines))

        for tgt in targets:
            tgt.update_step(config.scan_period)

        remove_threshold = config.max_range * config.remove_beyond_range_factor
        if config.dynamic_vessels:
            targets = [t for t in targets if np.linalg.norm(t.pos) <= remove_threshold]
        else:
            for tgt in targets:
                if np.linalg.norm(tgt.pos) > remove_threshold:
                    tgt.reset_randomly()

    plt.close(fig)

    return img_dir, lbl_dir


def generate_trails_dataset(
    src_img_dir,
    src_lbl_dir,
    output_dir,
    n_previous=5,
    alpha_decay=0.7,
    frame_fmt="frame_{:04d}.png",
    overwrite=True,
    preload=True,
    num_workers=0,
    black_threshold=10,
):
    """
    Generate a trails dataset from an existing radar dataset.

    - Images are composites of previous frames (trails) + current frame (red).
    - Labels are copied from the current frame.
    """
    img_dir, lbl_dir = _prepare_output_dirs(output_dir, overwrite)

    label_paths = sorted(Path(src_lbl_dir).glob("frame_*.txt"))
    if not label_paths:
        raise FileNotFoundError(f"No label files found in {src_lbl_dir}")

    frame_indices = []
    for lbl_path in label_paths:
        try:
            frame_idx = int(lbl_path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        frame_indices.append(frame_idx)

    if not frame_indices:
        raise ValueError("No valid frame indices found in labels.")

    max_idx = max(frame_indices)
    img_paths = [Path(src_img_dir) / frame_fmt.format(i) for i in range(max_idx + 1)]
    if not img_paths[0].exists():
        raise FileNotFoundError(f"Missing image: {img_paths[0]}")

    skipped = 0

    if preload:
        # Preload all frames into memory for speed.
        first_img = Image.open(img_paths[0]).convert("RGB")
        h_px, w_px = first_img.size[1], first_img.size[0]
        frames_u8 = np.zeros((max_idx + 1, h_px, w_px, 3), dtype=np.uint8)
        for i, p in enumerate(tqdm(img_paths, desc="Loading frames", unit="frame")):
            if p.exists():
                frames_u8[i] = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
            else:
                skipped += 1

        if num_workers > 0:
            shm = shared_memory.SharedMemory(create=True, size=frames_u8.nbytes)
            shm_arr = np.ndarray(frames_u8.shape, dtype=frames_u8.dtype, buffer=shm.buf)
            shm_arr[:] = frames_u8[:]

            args_list = [
                (
                    shm.name,
                    frames_u8.shape,
                    frames_u8.dtype,
                    frame_idx,
                    n_previous,
                    alpha_decay,
                    black_threshold,
                )
                for frame_idx in frame_indices
            ]

            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                futures = [ex.submit(_worker_build_trail, args) for args in args_list]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Compositing", unit="frame"):
                    frame_idx, out = fut.result()
                    out_img_path = os.path.join(img_dir, frame_fmt.format(frame_idx))
                    Image.fromarray(out).save(out_img_path)

            shm.close()
            shm.unlink()
        else:
            for frame_idx in tqdm(frame_indices, desc="Compositing", unit="frame"):
                out = _compose_trails_from_frames(
                    frames_u8=frames_u8,
                    frame_idx=frame_idx,
                    n_previous=n_previous,
                    alpha_decay=alpha_decay,
                    black_threshold=black_threshold,
                )
                out_img_path = os.path.join(img_dir, frame_fmt.format(frame_idx))
                Image.fromarray(out).save(out_img_path)
    else:
        for lbl_path in tqdm(label_paths, desc="Generating trails dataset", unit="frame"):
            try:
                frame_idx = int(lbl_path.stem.split("_")[1])
            except (IndexError, ValueError):
                skipped += 1
                continue

            composite = build_trails_composite(
                frame_idx=frame_idx,
                img_dir=src_img_dir,
                n_previous=n_previous,
                alpha_decay=alpha_decay,
                frame_fmt=frame_fmt,
            )
            if composite is None:
                skipped += 1
                continue

            out_img_path = os.path.join(img_dir, frame_fmt.format(frame_idx))
            composite.save(out_img_path)

    # Copy labels for the current frame.
    for lbl_path in label_paths:
        out_lbl_path = os.path.join(lbl_dir, lbl_path.name)
        shutil.copy2(lbl_path, out_lbl_path)

    if skipped:
        print(f"Skipped {skipped} frames due to missing/invalid inputs.")

    return img_dir, lbl_dir


def visualize_bboxes(image_path, label_path, title=None):
    img = Image.open(image_path)
    w_px, h_px = img.size

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)
    if title:
        ax.set_title(title)

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        cls, cx, cy, nw, nh = map(float, line.split())
        box_w = nw * w_px
        box_h = nh * h_px
        box_x = (cx * w_px) - (box_w / 2)
        box_y_visual = (cy * h_px) - (box_h / 2)

        rect = patches.Rectangle(
            (box_x, box_y_visual),
            box_w,
            box_h,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.show()
    return fig, ax


def verify_random_frame(img_dir, lbl_dir, num_frames, rng=None):
    rng = rng or np.random.default_rng()
    test_idx = int(rng.integers(0, num_frames))
    test_img_path = os.path.join(img_dir, f"frame_{test_idx:04d}.png")
    test_lbl_path = os.path.join(lbl_dir, f"frame_{test_idx:04d}.txt")

    if os.path.exists(test_img_path) and os.path.exists(test_lbl_path):
        visualize_bboxes(
            test_img_path,
            test_lbl_path,
            title=f"Verification: Frame {test_idx}",
        )
        print("Check the pop-up window to verify bounding box accuracy.")
    else:
        print("Verification frame not found.")
