import os

import cv2
import numpy as np


def _top_edge_stats(area_mask, image_width):
    ys, xs = np.where(area_mask)
    if len(xs) == 0:
        return {
            "width_coverage": 0.0,
            "top_edge_fill_ratio": 0.0,
            "top_edge_roughness_px": 0.0,
        }

    unique_x = np.unique(xs)
    width_coverage = len(unique_x) / max(image_width, 1)

    x_min = int(xs.min())
    x_max = int(xs.max())
    bbox_width = max(1, x_max - x_min + 1)
    top_edge_fill_ratio = len(unique_x) / bbox_width

    top_y_by_x = []
    for x in unique_x:
        column_ys = ys[xs == x]
        top_y_by_x.append(float(column_ys.min()))

    if len(top_y_by_x) < 2:
        roughness = 0.0
    else:
        roughness = float(np.median(np.abs(np.diff(top_y_by_x))))

    return {
        "width_coverage": float(width_coverage),
        "top_edge_fill_ratio": float(top_edge_fill_ratio),
        "top_edge_roughness_px": roughness,
    }


def _build_region_stats(mask, depth16, include_zero=False):
    height, width = mask.shape
    region_ids = np.unique(mask)
    stats = []

    for region_id in region_ids:
        if region_id < 0:
            continue
        if region_id == 0 and not include_zero:
            continue

        area_mask = mask == region_id
        if not np.any(area_mask):
            continue

        ys, xs = np.where(area_mask)
        area = int(len(xs))
        edge_stats = _top_edge_stats(area_mask, width)
        top = int(ys.min())
        bottom = int(ys.max())
        left = int(xs.min())
        right = int(xs.max())
        bbox_width_px = max(1, right - left + 1)
        bbox_height_px = max(1, bottom - top + 1)
        bottom_band_start = max(top, bottom - int(height * 0.10))
        bottom_band_xs = np.unique(xs[ys >= bottom_band_start])

        stats.append({
            "id": int(region_id),
            "avg_depth": float(np.mean(depth16[area_mask])),
            "area": area,
            "area_ratio": float(area / max(height * width, 1)),
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "bbox_width_ratio": float(bbox_width_px / max(width, 1)),
            "bbox_height_ratio": float(bbox_height_px / max(height, 1)),
            "verticality_ratio": float(bbox_height_px / bbox_width_px),
            "bottom_band_coverage": float(len(bottom_band_xs) / max(width, 1)),
            "bottom_ratio": float(bottom / max(height - 1, 1)),
            **edge_stats,
        })

    return stats


def _assign_depth_layers(region_stats, max_layers):
    layers = [[] for _ in range(max_layers)]
    if not region_stats:
        return layers

    depths = np.array([entry["avg_depth"] for entry in region_stats], dtype=np.float32)
    if len(depths) == 1:
        layers[0].append(region_stats[0])
        return layers

    thresholds = np.quantile(
        depths,
        [idx / max_layers for idx in range(1, max_layers)],
    )

    for entry in region_stats:
        layer_idx = int(np.searchsorted(thresholds, entry["avg_depth"], side="right"))
        layer_idx = int(np.clip(layer_idx, 0, max_layers - 1))
        layers[layer_idx].append(entry)

    return layers


def _score_track_candidate(
    entry,
    min_depth,
    max_track_depth,
    roughness_limit_px,
    max_verticality_ratio,
    target_area_ratio,
    target_height_ratio,
):
    if max_track_depth <= min_depth:
        depth_score = 1.0
    else:
        depth_score = 1.0 - (entry["avg_depth"] - min_depth) / (max_track_depth - min_depth)
        depth_score = float(np.clip(depth_score, 0.0, 1.0))

    roughness_score = 1.0 - entry["top_edge_roughness_px"] / max(roughness_limit_px, 1.0)
    roughness_score = float(np.clip(roughness_score, 0.0, 1.0))
    verticality_score = 1.0 - entry["verticality_ratio"] / max(max_verticality_ratio, 1e-6)
    verticality_score = float(np.clip(verticality_score, 0.0, 1.0))
    area_score = 1.0 - max(0.0, entry["area_ratio"] - target_area_ratio) / max(target_area_ratio, 1e-6)
    area_score = float(np.clip(area_score, 0.0, 1.0))
    height_score = 1.0 - max(0.0, entry["bbox_height_ratio"] - target_height_ratio) / max(target_height_ratio, 1e-6)
    height_score = float(np.clip(height_score, 0.0, 1.0))

    return (
        entry["bottom_ratio"] * 2.0
        + entry["width_coverage"] * 2.0
        + entry["bottom_band_coverage"] * 2.0
        + min(entry["area_ratio"] * 20.0, 1.0)
        + entry["top_edge_fill_ratio"]
        + depth_score
        + roughness_score
        + verticality_score
        + area_score
        + height_score
    )


def _select_track_regions(
    region_stats,
    height,
    min_bottom_ratio,
    min_width_coverage,
    min_area_ratio,
    min_top_edge_fill_ratio,
    max_top_edge_roughness_ratio,
    max_track_depth_quantile,
    min_bottom_band_coverage,
    max_verticality_ratio,
    max_area_ratio,
    max_bbox_height_ratio,
    target_area_ratio,
    target_height_ratio,
    max_regions,
    max_combined_area_ratio,
):
    if not region_stats:
        print("No SAM2 regions available for track selection; writing empty track mask.")
        return []

    depths = np.array([entry["avg_depth"] for entry in region_stats], dtype=np.float32)
    min_depth = float(depths.min())
    max_track_depth = float(np.quantile(depths, max_track_depth_quantile))
    roughness_limit_px = float(height * max_top_edge_roughness_ratio)

    strict_candidates = []
    scored = []
    print("\nTrack region candidates:")
    for entry in region_stats:
        score = _score_track_candidate(
            entry,
            min_depth,
            max_track_depth,
            roughness_limit_px,
            max_verticality_ratio,
            target_area_ratio,
            target_height_ratio,
        )
        scored.append((score, entry))

        depth_quantile = float(np.mean(depths <= entry["avg_depth"]))
        roughness_ratio = float(entry["top_edge_roughness_px"] / max(height, 1))
        reject_reasons = []

        if entry["bottom_ratio"] < min_bottom_ratio:
            reject_reasons.append(f"bottom<{min_bottom_ratio:.2f}")
        if entry["width_coverage"] < min_width_coverage:
            reject_reasons.append(f"width<{min_width_coverage:.2f}")
        if entry["area_ratio"] < min_area_ratio:
            reject_reasons.append(f"area<{min_area_ratio:.3f}")
        if entry["area_ratio"] > max_area_ratio:
            reject_reasons.append(f"area>{max_area_ratio:.3f}")
        if entry["bbox_height_ratio"] > max_bbox_height_ratio:
            reject_reasons.append(f"height>{max_bbox_height_ratio:.2f}")
        if entry["top_edge_fill_ratio"] < min_top_edge_fill_ratio:
            reject_reasons.append(f"fill<{min_top_edge_fill_ratio:.2f}")
        if entry["top_edge_roughness_px"] > roughness_limit_px:
            reject_reasons.append(f"roughness>{max_top_edge_roughness_ratio:.3f}")
        if entry["avg_depth"] > max_track_depth:
            reject_reasons.append(f"depth_q>{max_track_depth_quantile:.2f}")
        if entry["bottom_band_coverage"] < min_bottom_band_coverage:
            reject_reasons.append(f"bottom_band<{min_bottom_band_coverage:.2f}")
        if entry["verticality_ratio"] > max_verticality_ratio:
            reject_reasons.append(f"verticality>{max_verticality_ratio:.2f}")

        passed = not reject_reasons
        if passed:
            strict_candidates.append(entry)

        print(
            f"  id={entry['id']} "
            f"bottom={entry['bottom_ratio']:.3f} "
            f"width={entry['width_coverage']:.3f} "
            f"area={entry['area_ratio']:.3f} "
            f"height={entry['bbox_height_ratio']:.3f} "
            f"bottom_band={entry['bottom_band_coverage']:.3f} "
            f"verticality={entry['verticality_ratio']:.2f} "
            f"fill={entry['top_edge_fill_ratio']:.3f} "
            f"roughness={roughness_ratio:.3f} "
            f"depth_q={depth_quantile:.2f} "
            f"mean_depth={entry['avg_depth']:.1f} "
            f"score={score:.2f} "
            f"{'PASS' if passed else 'REJECT ' + ','.join(reject_reasons)}"
        )

    if strict_candidates:
        strict_candidates.sort(
            key=lambda entry: _score_track_candidate(
                entry,
                min_depth,
                max_track_depth,
                roughness_limit_px,
                max_verticality_ratio,
                target_area_ratio,
                target_height_ratio,
            ),
            reverse=True,
        )

        selected = []
        combined_area = 0.0
        for entry in strict_candidates:
            if len(selected) >= max_regions:
                break
            if selected and combined_area + entry["area_ratio"] > max_combined_area_ratio:
                continue
            selected.append(entry)
            combined_area += entry["area_ratio"]

        if not selected:
            selected = strict_candidates[:1]

        print(
            f"Track selection: using {len(selected)} of {len(strict_candidates)} "
            "strict candidate(s) after size cap."
        )
        return sorted(selected, key=lambda item: item["left"])

    scored.sort(key=lambda item: item[0], reverse=True)
    fallback = [
        entry for _, entry in scored
        if entry["bottom_ratio"] >= max(0.0, min_bottom_ratio - 0.10)
        and entry["width_coverage"] >= min_width_coverage * 0.5
        and entry["area_ratio"] >= min_area_ratio * 0.5
        and entry["area_ratio"] <= max_area_ratio
        and entry["bbox_height_ratio"] <= max_bbox_height_ratio * 1.15
        and entry["bottom_band_coverage"] >= min_bottom_band_coverage * 0.5
        and entry["top_edge_fill_ratio"] >= min_top_edge_fill_ratio * 0.7
        and entry["top_edge_roughness_px"] <= roughness_limit_px * 1.5
        and entry["avg_depth"] <= float(np.quantile(depths, min(0.95, max_track_depth_quantile + 0.10)))
        and entry["verticality_ratio"] <= max_verticality_ratio * 1.35
    ]

    if fallback:
        print(
            "Track selection fallback: no strict candidate passed; "
            f"using {len(fallback[:max_regions])} relaxed candidate(s)."
        )
        selected = []
        combined_area = 0.0
        for entry in fallback:
            if len(selected) >= max_regions:
                break
            if selected and combined_area + entry["area_ratio"] > max_combined_area_ratio:
                continue
            selected.append(entry)
            combined_area += entry["area_ratio"]
        return sorted(selected, key=lambda item: item["left"])

    bottom_area_fallback = max(
        region_stats,
        key=lambda entry: (
            entry["bottom_ratio"] * 2.0
            + min(entry["area_ratio"] * 20.0, 1.0)
            + entry["width_coverage"]
        ),
    )
    print(
        "Track selection fallback: no relaxed candidate passed; "
        f"using largest bottom-near region id={bottom_area_fallback['id']}."
    )
    return [bottom_area_fallback]


def _combined_mask_for_regions(mask, regions):
    combined_mask = np.zeros(mask.shape, dtype=np.uint8)
    for entry in regions:
        combined_mask[mask == entry["id"]] = 255
    return combined_mask


def process_cut_img(
    image_path,
    depth16_path,
    mask_path,
    out_dir="./Send2Unity",
    max_layers=3,
    dilate_pixels=5,
    feather_pixels=2,
    inpaint_holes=True,
    track_min_bottom_ratio=0.58,
    track_min_width_coverage=0.28,
    track_min_area_ratio=0.012,
    track_min_top_edge_fill_ratio=0.50,
    track_max_top_edge_roughness_ratio=0.055,
    track_max_depth_quantile=0.82,
    track_min_bottom_band_coverage=0.16,
    track_max_verticality_ratio=1.45,
    track_max_area_ratio=0.32,
    track_max_bbox_height_ratio=0.62,
    track_target_area_ratio=0.18,
    track_target_height_ratio=0.45,
    track_max_regions=2,
    track_max_combined_area_ratio=0.34,
):
    rgb_img = cv2.imread(image_path)
    if rgb_img is None:
        print(f"Failed to load image: {image_path}")
        return
    rgba_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2BGRA)

    depth16 = cv2.imread(depth16_path, cv2.IMREAD_UNCHANGED)
    if depth16 is None:
        print(f"Failed to load depth image: {depth16_path}")
        return

    if not os.path.exists(mask_path):
        print(f"Failed to find mask: {mask_path}")
        return

    mask = np.load(mask_path).copy()
    height, width = mask.shape

    if np.any(mask == 0):
        kernel = np.ones((5, 5), np.uint8)
        mask_0 = (mask == 0).astype(np.uint8)
        eroded_0 = cv2.erode(mask_0, kernel, iterations=2)
        to_be_inpainted = (mask_0 == 1) & (eroded_0 == 0)
        mask[to_be_inpainted] = -1

    visual_stats = _build_region_stats(mask, depth16, include_zero=True)
    track_stats = _build_region_stats(mask, depth16, include_zero=False)
    visual_layers = _assign_depth_layers(visual_stats, max_layers)
    track_regions = _select_track_regions(
        track_stats,
        height,
        track_min_bottom_ratio,
        track_min_width_coverage,
        track_min_area_ratio,
        track_min_top_edge_fill_ratio,
        track_max_top_edge_roughness_ratio,
        track_max_depth_quantile,
        track_min_bottom_band_coverage,
        track_max_verticality_ratio,
        track_max_area_ratio,
        track_max_bbox_height_ratio,
        track_target_area_ratio,
        track_target_height_ratio,
        track_max_regions,
        track_max_combined_area_ratio,
    )
    track_mask = _combined_mask_for_regions(mask, track_regions)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "layer_00_mask.npy"), track_mask)
    cv2.imwrite(os.path.join(out_dir, "track_mask.png"), track_mask)

    print(f"Merging visual output into {max_layers} depth layers...")
    print(
        "Selected track regions: "
        + ", ".join(
            f"id={entry['id']} bottom={entry['bottom_ratio']:.2f} "
            f"width={entry['width_coverage']:.2f} area={entry['area_ratio']:.3f}"
            for entry in track_regions
        )
    )

    for layer_idx, layer_regions in enumerate(visual_layers):
        combined_mask = _combined_mask_for_regions(mask, layer_regions)

        if layer_idx == max_layers - 1 and inpaint_holes:
            print("  Inpainting background holes...")
            hole_mask = cv2.bitwise_not(combined_mask)
            inpainted_bgr = cv2.inpaint(rgb_img, hole_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            cut_img = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2BGRA)
            cut_img[:, :, 3] = 255
        else:
            if dilate_pixels > 0:
                kernel = np.ones((dilate_pixels, dilate_pixels), np.uint8)
                alpha_mask = cv2.dilate(combined_mask, kernel, iterations=1)
            else:
                alpha_mask = combined_mask

            if feather_pixels > 0:
                alpha_mask = cv2.GaussianBlur(alpha_mask, (0, 0), sigmaX=feather_pixels / 3.0)

            cut_img = rgba_img.copy()
            cut_img[:, :, 3] = alpha_mask

        filename = f"merged_layer_{layer_idx + 1:02d}.png"
        save_path = os.path.join(out_dir, filename)
        cv2.imwrite(save_path, cut_img)

        pos_desc = "NEAR" if layer_idx == 0 else "FAR(Inpainted)" if layer_idx == max_layers - 1 else "MID"
        print(f"  Saved {filename} ({pos_desc}, Contains {len(layer_regions)} regions)")

    print("\n--- Done. Visual layers and dedicated track mask saved. ---")


if __name__ == "__main__":
    base_name = "street2d"

    rgb_img_path = f"./assets/examples/SOH/{base_name}.jpg"
    depth16_path_arg = f"./outputs/inference_results/{base_name}_depth_16bit.png"
    mask_npy_path = f"./outputs/inference_results/{base_name}_depth_mask.npy"

    process_cut_img(rgb_img_path, depth16_path_arg, mask_npy_path)
