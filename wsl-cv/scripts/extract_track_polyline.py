import datetime
import json
import os

import cv2
import numpy as np


def _clip_point(x, y, width, height):
    x = float(np.clip(x, 0, width - 1))
    y = float(np.clip(y, 0, height - 1))
    return x, y


def _clip_norm(value):
    return round(float(np.clip(value, 0.0, 1.0)), 4)


def _extract_top_edge(mask):
    height, width = mask.shape
    y_values = np.full(width, np.nan, dtype=np.float32)

    for x in range(width):
        indices = np.where(mask[:, x] > 127)[0]
        if len(indices) > 0:
            y_values[x] = float(indices[0])

    valid = np.isfinite(y_values)
    if not np.any(valid):
        y_values[:] = height * 0.75
        return y_values

    x_all = np.arange(width, dtype=np.float32)
    x_valid = x_all[valid]
    y_valid = y_values[valid]

    return np.interp(
        x_all,
        x_valid,
        y_valid,
        left=float(y_valid[0]),
        right=float(y_valid[-1]),
    ).astype(np.float32)


def _apply_median_filter(y_values, median_size):
    if median_size is None or median_size <= 1:
        return y_values

    kernel_size = int(median_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    padded = np.pad(y_values, kernel_size // 2, mode="edge")
    filtered = np.empty_like(y_values, dtype=np.float32)
    for idx in range(len(y_values)):
        filtered[idx] = np.median(padded[idx : idx + kernel_size])
    return filtered


def _simplify_edge(y_values, width, height, epsilon):
    edge_points = np.array(
        [[[x, int(round(y))]] for x, y in enumerate(y_values)],
        dtype=np.int32,
    )
    simplified = cv2.approxPolyDP(edge_points, float(epsilon), False)
    key_points = [(float(pt[0][0]), float(pt[0][1])) for pt in simplified]

    if not key_points:
        key_points = [(0.0, float(y_values[0])), (float(width - 1), float(y_values[-1]))]

    key_points.append((0.0, float(y_values[0])))
    key_points.append((float(width - 1), float(y_values[-1])))

    by_x = {}
    for x, y in key_points:
        x, y = _clip_point(x, y, width, height)
        by_x[x] = y

    return [(x, by_x[x]) for x in sorted(by_x)]


def _segment_angle_degrees(start, end):
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    if abs(dx) < 1e-6:
        return 90.0 if dy >= 0 else -90.0
    return float(np.degrees(np.arctan2(dy, dx)))


def _point_distance(start, end):
    return float(np.hypot(end[0] - start[0], end[1] - start[1]))


def _corner_angle_degrees(prev_point, corner_point, next_point):
    prev_vec = np.array(prev_point, dtype=np.float32) - np.array(corner_point, dtype=np.float32)
    next_vec = np.array(next_point, dtype=np.float32) - np.array(corner_point, dtype=np.float32)
    prev_len = float(np.linalg.norm(prev_vec))
    next_len = float(np.linalg.norm(next_vec))
    if prev_len < 1e-6 or next_len < 1e-6:
        return 180.0

    cos_angle = float(np.dot(prev_vec, next_vec) / (prev_len * next_len))
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _line_y_at_x(start, end, x):
    if abs(end[0] - start[0]) < 1e-6:
        return (start[1] + end[1]) * 0.5

    t = (x - start[0]) / (end[0] - start[0])
    return start[1] + (end[1] - start[1]) * t


def _project_keypoint_to_bridge(prev_point, point, next_point, width, height):
    projected_y = _line_y_at_x(prev_point, next_point, point[0])
    return _clip_point(point[0], projected_y, width, height)


def cleanup_keypoints(
    key_points,
    width,
    height,
    max_segment_angle_degrees=55.0,
    min_corner_angle_degrees=105.0,
    valley_depth_threshold_px=35.0,
    peak_height_threshold_px=30.0,
    min_segment_length_px=25.0,
):
    if len(key_points) < 3:
        return key_points

    max_angle = abs(float(max_segment_angle_degrees))
    min_corner_angle = float(min_corner_angle_degrees)
    valley_threshold = float(valley_depth_threshold_px)
    peak_threshold = float(peak_height_threshold_px)
    min_segment_length = float(min_segment_length_px)

    points = [tuple(_clip_point(point[0], point[1], width, height)) for point in key_points]

    max_passes = max(4, len(points) * 2)
    for _ in range(max_passes):
        if len(points) < 3:
            break

        changed = False
        cleaned = [points[0]]
        idx = 1

        while idx < len(points) - 1:
            prev_point = cleaned[-1]
            point = points[idx]
            next_point = points[idx + 1]

            segment_in_angle = abs(_segment_angle_degrees(prev_point, point))
            segment_out_angle = abs(_segment_angle_degrees(point, next_point))
            bridge_angle = abs(_segment_angle_degrees(prev_point, next_point))
            corner_angle = _corner_angle_degrees(prev_point, point, next_point)
            bridge_y = _line_y_at_x(prev_point, next_point, point[0])

            has_short_segment = (
                _point_distance(prev_point, point) < min_segment_length
                or _point_distance(point, next_point) < min_segment_length
            )
            is_deep_valley = point[1] - bridge_y >= valley_threshold
            is_sharp_peak = bridge_y - point[1] >= peak_threshold
            is_sharp_corner = corner_angle < min_corner_angle
            is_steep = segment_in_angle >= max_angle or segment_out_angle >= max_angle

            if has_short_segment or is_deep_valley or is_sharp_peak or is_sharp_corner:
                changed = True
                idx += 1
                continue

            if is_steep:
                if bridge_angle < max_angle:
                    changed = True
                    idx += 1
                    continue

                projected = _project_keypoint_to_bridge(prev_point, point, next_point, width, height)
                projected_in_angle = abs(_segment_angle_degrees(prev_point, projected))
                projected_out_angle = abs(_segment_angle_degrees(projected, next_point))

                if (
                    projected_in_angle < max_angle
                    and projected_out_angle < max_angle
                    and _point_distance(prev_point, projected) >= min_segment_length
                    and _point_distance(projected, next_point) >= min_segment_length
                ):
                    cleaned.append(projected)
                else:
                    idx += 1
                    changed = True
                    continue

                changed = changed or _point_distance(point, projected) > 1e-6
                idx += 1
                continue

            cleaned.append(point)
            idx += 1

        cleaned.append(points[-1])

        if not changed:
            return cleaned
        points = cleaned

    return points


def _append_point(points, x, y, width, height):
    x, y = _clip_point(x, y, width, height)
    if points and abs(points[-1][0] - x) < 1e-6 and abs(points[-1][1] - y) < 1e-6:
        return
    points.append((x, y))


def _append_line(points, start, end, width, height, sample_interval):
    sx, sy = start
    ex, ey = end
    distance = float(np.hypot(ex - sx, ey - sy))
    steps = max(1, int(np.ceil(distance / max(float(sample_interval), 1.0))))
    for idx in range(steps + 1):
        t = idx / steps
        x = sx + (ex - sx) * t
        y = sy + (ey - sy) * t
        _append_point(points, x, y, width, height)


def _rounded_polyline_points(key_points, width, height, corner_radius_px, corner_samples, sample_interval):
    if len(key_points) <= 2 or corner_radius_px <= 0 or corner_samples <= 0:
        output = []
        for start, end in zip(key_points[:-1], key_points[1:]):
            _append_line(output, start, end, width, height, sample_interval)
        return output

    output = []
    current = key_points[0]
    _append_point(output, current[0], current[1], width, height)

    for idx in range(1, len(key_points) - 1):
        prev_pt = np.array(key_points[idx - 1], dtype=np.float32)
        corner = np.array(key_points[idx], dtype=np.float32)
        next_pt = np.array(key_points[idx + 1], dtype=np.float32)

        prev_vec = prev_pt - corner
        next_vec = next_pt - corner
        prev_len = float(np.linalg.norm(prev_vec))
        next_len = float(np.linalg.norm(next_vec))

        if prev_len < 1e-6 or next_len < 1e-6:
            _append_line(output, current, tuple(corner), width, height, sample_interval)
            current = tuple(corner)
            continue

        radius = min(float(corner_radius_px), prev_len * 0.5, next_len * 0.5)
        entry = corner + (prev_vec / prev_len) * radius
        exit_pt = corner + (next_vec / next_len) * radius

        _append_line(output, current, tuple(entry), width, height, sample_interval)

        samples = max(2, int(corner_samples))
        for sample_idx in range(1, samples + 1):
            t = sample_idx / samples
            one_minus_t = 1.0 - t
            bezier = (
                one_minus_t * one_minus_t * entry
                + 2.0 * one_minus_t * t * corner
                + t * t * exit_pt
            )
            _append_point(output, bezier[0], bezier[1], width, height)

        current = tuple(exit_pt)

    _append_line(output, current, key_points[-1], width, height, sample_interval)
    return output


def _track_color(rgb_image_path, pixel_points, width, height):
    track_color_hex = "#9e5752"
    if not rgb_image_path or not os.path.exists(rgb_image_path):
        return track_color_hex

    rgb_img = cv2.imread(rgb_image_path)
    if rgb_img is None:
        return track_color_hex

    img_h, img_w = rgb_img.shape[:2]
    colors = []
    for x, y in pixel_points:
        px = int(round((x / max(width - 1, 1)) * (img_w - 1)))
        py = int(round((y / max(height - 1, 1)) * (img_h - 1)))
        px = int(np.clip(px, 0, img_w - 1))
        py = int(np.clip(py, 0, img_h - 1))

        y_start = max(0, py - 2)
        y_end = min(img_h, py + 3)
        x_start = max(0, px - 2)
        x_end = min(img_w, px + 3)
        region = rgb_img[y_start:y_end, x_start:x_end]
        if region.size > 0:
            colors.append(np.mean(region, axis=(0, 1)))

    if colors:
        mean_bgr = np.mean(colors, axis=0)
        b, g, r = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])
        track_color_hex = f"#{r:02x}{g:02x}{b:02x}"

    return track_color_hex


def _write_visualization(mask, y_values, key_points, pixel_points, mask_npy_path):
    height, width = mask.shape
    vis_img = np.zeros((height, width, 3), dtype=np.uint8)
    vis_img[mask > 127] = (50, 50, 50)

    for x in range(width - 1):
        cv2.line(
            vis_img,
            (x, int(round(y_values[x]))),
            (x + 1, int(round(y_values[x + 1]))),
            (0, 0, 255),
            1,
        )

    for x, y in key_points:
        cv2.circle(vis_img, (int(round(x)), int(round(y))), 4, (0, 255, 255), -1)

    for start, end in zip(pixel_points[:-1], pixel_points[1:]):
        cv2.line(
            vis_img,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            (0, 255, 0),
            2,
        )

    vis_path = mask_npy_path.replace(".npy", "_polyline_track_vis.png")
    cv2.imwrite(vis_path, vis_img)
    print(f"Saved visualization to {vis_path}")


def extract_polyline_track(
    mask_npy_path,
    out_json_path,
    rgb_image_path=None,
    source_img_name=None,
    epsilon=16.0,
    median_size=9,
    corner_radius_px=16.0,
    corner_samples=6,
    sample_interval=12.0,
    max_segment_angle_degrees=55.0,
    min_corner_angle_degrees=105.0,
    valley_depth_threshold_px=35.0,
    peak_height_threshold_px=30.0,
    min_segment_length_px=25.0,
):
    if not os.path.exists(mask_npy_path):
        print(f"Error: {mask_npy_path} not found.")
        return

    print(f"Loading {mask_npy_path}...")
    mask = np.load(mask_npy_path)
    height, width = mask.shape

    y_values = _extract_top_edge(mask)
    y_values = _apply_median_filter(y_values, median_size)
    y_values = np.clip(y_values, 0, height - 1)

    key_points = _simplify_edge(y_values, width, height, epsilon)
    key_points = cleanup_keypoints(
        key_points,
        width,
        height,
        max_segment_angle_degrees,
        min_corner_angle_degrees,
        valley_depth_threshold_px,
        peak_height_threshold_px,
        min_segment_length_px,
    )
    pixel_points = _rounded_polyline_points(
        key_points,
        width,
        height,
        corner_radius_px,
        corner_samples,
        sample_interval,
    )

    if len(pixel_points) < 2:
        y_fallback = float(np.clip(height * 0.75, 0, height - 1))
        pixel_points = [(0.0, y_fallback), (float(width - 1), y_fallback)]

    track_points = []
    for x, y in pixel_points:
        x, y = _clip_point(x, y, width, height)
        track_points.append([
            _clip_norm(x / max(width - 1, 1)),
            _clip_norm(y / max(height - 1, 1)),
        ])

    json_data = {
        "version": "1.0",
        "timestamp": datetime.datetime.utcnow().isoformat("T", "microseconds") + "Z",
        "source_image": source_img_name or os.path.basename(rgb_image_path or "image.jpg"),
        "aspect_ratio": round(width / height, 4),
        "track_color": _track_color(rgb_image_path, pixel_points, width, height),
        "points": track_points,
    }

    _write_visualization(mask, y_values, key_points, pixel_points, mask_npy_path)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    print(f"Successfully generated polyline track with {len(track_points)} points.")
    print(f"Saved JSON to {out_json_path}")


if __name__ == "__main__":
    npy_file = "./Send2Unity/layer_00_mask.npy"
    out_json = "./Send2Unity/track_points.json"
    original_img_path = "./assets/examples/SOH/street2d.jpg"

    extract_polyline_track(
        npy_file,
        out_json,
        rgb_image_path=original_img_path,
        source_img_name=os.path.basename(original_img_path),
    )
