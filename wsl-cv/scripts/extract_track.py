import numpy as np
import cv2
import json
import os
import numpy as np
import cv2
import scipy.ndimage
from scipy.interpolate import CubicSpline
import datetime

def _clip_pixel_point(x, y, width, height, top_margin_px=0.0):
    x = float(np.clip(x, 0, width - 1))
    y_min = float(np.clip(top_margin_px, 0, max(height - 1, 0)))
    y = float(np.clip(y, y_min, height - 1))
    return x, y

def _clip_norm(value):
    return round(float(np.clip(value, 0.0, 1.0)), 4)

def _clip_and_smooth_y(y_values, height, sigma, top_margin_px=0.0):
    y_min = float(np.clip(top_margin_px, 0, max(height - 1, 0)))
    y_values = np.clip(y_values, y_min, height - 1)
    if sigma and sigma > 0 and len(y_values) > 2:
        y_values = scipy.ndimage.gaussian_filter1d(y_values, sigma=sigma, mode="nearest")
        y_values = np.clip(y_values, y_min, height - 1)
    return y_values

def extract_smooth_track(npy_path, output_json, rgb_image_path=None, smooth_sigma=30.0, epsilon=20.0, alpha=0.5, sample_interval=20.0, source_img_name="image.jpg", clip_smooth_sigma=1.5, clip_top_margin_px=50.0):
    """
    smooth_sigma: 高斯平滑強度 (值越大，原本鋸齒狀的地形會被撫得越平，小碎石都會消失)
    epsilon: 多邊形簡化誤差容忍度 (值越大，抓出的「轉折關鍵點」越少，線段會變得很長、很剛硬)
    alpha: 0.0 代表完全是一條直線，1.0 代表完全貼合平滑後的邊緣
    sample_interval: 貝茲曲線採樣間距 (數值越大，最後輸出的 JSON 軌道點越少)
    clip_smooth_sigma: 對已限制在圖片內的輸出曲線做小範圍平滑，降低硬 clipping 的鋸齒感
    clip_top_margin_px: 上方 clipping 邊界，避免軌道貼到圖片最上緣
    """
    if not os.path.exists(npy_path):
        print(f"Error: {npy_path} not found.")
        return

    print(f"Loading {npy_path}...")
    mask = np.load(npy_path)
    h, w = mask.shape
    
    # Extract the highest pixel (min y) for each x coordinate
    y_values = []
    
    for x in range(w):
        column = mask[:, x]
        indices = np.where(column > 127)[0]
        if len(indices) > 0:
            y_values.append(indices[0]) # The topmost pixel in this column
        else:
            # If no pixels in this column, just reuse the previous Y or set to bottom
            if len(y_values) > 0:
                y_values.append(y_values[-1])
            else:
                y_values.append(h - 1)
                
    y_values = np.array(y_values)
    
    # 找出最符合真實邊緣的「最佳擬合直線」(Linear Regression)
    # 這條線代表整個地形的總體坡度趨勢，不再強迫必須是純水平 (斜率=0)
    x_indices = np.arange(h_w_len) if 'h_w_len' in locals() else np.arange(len(y_values))
    m, c = np.polyfit(x_indices, y_values, 1)
    best_fit_line = m * x_indices + c
    
    # 對真實邊緣進行適度的高斯平滑
    y_smoothed = scipy.ndimage.gaussian_filter1d(y_values, sigma=smooth_sigma)
    
    # 將真實平滑邊緣與地形總趨勢線進行插值 (Interpolation)
    # 這樣只要照片是有下坡或上坡，軌道就會趨近於該坡度，而不再死板地拉平
    y_interp = alpha * y_smoothed + (1.0 - alpha) * best_fit_line
    
    # Convert to standard point format
    points = np.array([ [[x, int(y)]] for x, y in enumerate(y_interp)], dtype=np.int32)
    
    # 將軌道簡化成直連的線段 (Piecewise linear，主要用來抓取轉折關鍵點)
    simplified_points = cv2.approxPolyDP(points, epsilon, False)
    
    # 提取這些關鍵點
    x_keys = []
    y_keys = []
    for pt in simplified_points:
        x, y = _clip_pixel_point(pt[0][0], pt[0][1], w, h, clip_top_margin_px)
        x_keys.append(x)
        y_keys.append(y)
        
    # 確保 x_keys 是遞增且不重複的，才能做樣條插值
    x_keys, unique_idx = np.unique(x_keys, return_index=True)
    y_keys = np.array(y_keys)[unique_idx]
    
    # 排序 X 座標
    sort_idx = np.argsort(x_keys)
    x_keys = x_keys[sort_idx]
    y_keys = y_keys[sort_idx]
    
    # Prepare JSON data: 套用貝茲曲線級別的平滑 (Cubic Spline)
    track_points = []
    if len(x_keys) > 2:
        # 使用三次方樣條插值產生完美平滑的曲線
        cs = CubicSpline(x_keys, y_keys, bc_type='natural')
        # 在頭尾兩點之間，依據 sample_interval 提取平滑軌道點
        num_samples = max(10, int((x_keys[-1] - x_keys[0]) / sample_interval))
        x_smooth = np.linspace(x_keys[0], x_keys[-1], num_samples)
        y_smooth = cs(x_smooth)
        y_smooth = _clip_and_smooth_y(y_smooth, h, clip_smooth_sigma, clip_top_margin_px)
        
        for x, y in zip(x_smooth, y_smooth):
            x, y = _clip_pixel_point(x, y, w, h, clip_top_margin_px)
            # 將座標正規化為 0.0 ~ 1.0，並四捨五入到小數點下四位
            x_norm = _clip_norm(float(x) / max(w - 1, 1))
            y_norm = _clip_norm(float(y) / max(h - 1, 1))
            track_points.append([x_norm, y_norm])
    else:
        # 如果點太少(只有兩個點)，就直接用原本的直線
        for x, y in zip(x_keys, y_keys):
            x, y = _clip_pixel_point(x, y, w, h, clip_top_margin_px)
            x_norm = _clip_norm(float(x) / max(w - 1, 1))
            y_norm = _clip_norm(float(y) / max(h - 1, 1))
            track_points.append([x_norm, y_norm])
            
    # 計算軌道經過區域的平均顏色
    track_color_hex = "#9e5752" # 預設值
    if rgb_image_path and os.path.exists(rgb_image_path):
        rgb_img = cv2.imread(rgb_image_path) # BGR format
        if rgb_img is not None:
            # 確保圖片大小跟 mask 是一致的
            img_h, img_w = rgb_img.shape[:2]
            colors = []
            
            for pt in track_points:
                px = int(np.clip(pt[0], 0.0, 1.0) * (img_w - 1))
                py = int(np.clip(pt[1], 0.0, 1.0) * (img_h - 1))
                
                # 為了避免拿到太多雜訊，擷取軌道點周圍 5x5 的小區域計算平均色
                y_start = max(0, py - 2)
                y_end = min(img_h, py + 3)
                x_start = max(0, px - 2)
                x_end = min(img_w, px + 3)
                
                region = rgb_img[y_start:y_end, x_start:x_end]
                if region.size > 0:
                    colors.append(np.mean(region, axis=(0,1)))
            
            if len(colors) > 0:
                mean_bgr = np.mean(colors, axis=0) # [B, G, R]
                b, g, r = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])
                track_color_hex = f"#{r:02x}{g:02x}{b:02x}"
            
    # 組裝成符合 example.json 格式的字典
    json_data = {
        "version": "1.0",
        "timestamp": datetime.datetime.utcnow().isoformat("T", "microseconds") + "Z",
        "source_image": source_img_name,
        "aspect_ratio": round(w / h, 4),
        "track_color": track_color_hex,
        "points": track_points
    }
            
    # --- Visualization ---
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)
    vis_img[mask > 127] = (50, 50, 50) # Gray background for mask
    
    # Draw original unsmoothed points (red)
    for i in range(len(y_values)-1):
        cv2.line(vis_img, (i, int(y_values[i])), (i+1, int(y_values[i+1])), (0, 0, 255), 1)
        
    # Draw the best horizontal line (green line, yellow dots at ends)
    # 繪製插值且經過樣條平滑的最終貝茲路徑 (綠線)
    # 此處還原回 pixel 座標來畫圖
    for i in range(len(track_points)-1):
        pt1_x = int(np.clip(track_points[i][0], 0.0, 1.0) * (w - 1))
        pt1_y = int(np.clip(track_points[i][1], 0.0, 1.0) * (h - 1))
        pt2_x = int(np.clip(track_points[i+1][0], 0.0, 1.0) * (w - 1))
        pt2_y = int(np.clip(track_points[i+1][1], 0.0, 1.0) * (h - 1))
        cv2.line(vis_img, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 2)
        
    # 畫出做為算圖骨架的關鍵點 (黃點)
    for x, y in zip(x_keys, y_keys):
        x, y = _clip_pixel_point(x, y, w, h, clip_top_margin_px)
        cv2.circle(vis_img, (int(x), int(y)), 4, (0, 255, 255), -1)
    
    vis_path = npy_path.replace('.npy', '_track_vis.png')
    cv2.imwrite(vis_path, vis_img)
    print(f"Saved visualization to {vis_path}")

    # Save JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)
        
    print(f"Successfully generated track with {len(track_points)} points.")
    print(f"Saved JSON to {output_json}")

if __name__ == '__main__':
    npy_file = './Send2Unity/layer_00_mask.npy'
    out_json = './Send2Unity/track_points.json'
    
    # 指向真實的原圖路徑，用來提取平均色彩
    original_img_path = './assets/examples/SOH/street2d.jpg'
    img_name = os.path.basename(original_img_path)
    
    extract_smooth_track(npy_file, out_json, rgb_image_path=original_img_path, source_img_name=img_name)
