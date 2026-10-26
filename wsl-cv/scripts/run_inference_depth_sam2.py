# run_inference_depth_sam2.py
import os, glob, cv2
import numpy as np, torch
from depth_adapter import get_depth_model, predict_depth
from sam2_segmentation import generate_sam2_label_map

def colorize_labels(labels: np.ndarray, num_labels: int) -> np.ndarray:
    """Assigns random colors to each label id for visualization."""
    np.random.seed(42)  # For consistent colors across frames
    colors = np.random.randint(0, 255, size=(max(num_labels, 1), 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # edges/background as black
    return colors[labels]

def _resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if width == target_width:
        return image

    scale = target_width / float(width)
    target_height = max(1, int(height * scale))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def run_inference(
    image_path,
    out_dir="./outputs/inference_results",
    model_name="depth-anything/da3-small",
    input_size=768,
    save_16bit=True,
    target_width=1024,
    inference_width=768,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Preparing depth model {model_name} on {device}...")
    model = get_depth_model(model_name, device)

    # Prepare inputs
    img_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif')
    inputs = []
    if os.path.isfile(image_path):
        inputs = [image_path]
    else:
        paths = glob.glob(os.path.join(image_path, '**/*'), recursive=True)
        for p in paths:
            if os.path.splitext(p)[1].lower() in img_exts:
                inputs.append(p)

    if not inputs:
        print(f"No images found at {image_path}")
        return

    os.makedirs(out_dir, exist_ok=True)

    for idx, src in enumerate(inputs):
        print(f'[{idx+1}/{len(inputs)}] Processing {src}')
        
        source_img = cv2.imread(src)
        if source_img is None:
            print(f"Failed to read {src}")
            continue

        output_img = _resize_to_width(source_img, target_width)
        output_h, output_w = output_img.shape[:2]
        if output_img.shape[:2] != source_img.shape[:2]:
            # Downstream cut_img.py reads this same path, so standardize only
            # the Unity-facing source image size, not the model inference size.
            cv2.imwrite(src, output_img)
            print(f'  ✔ Resized output image to {output_w}x{output_h}')

        inference_img = _resize_to_width(source_img, inference_width)
        inference_h, inference_w = inference_img.shape[:2]
        inference_src = os.path.join(out_dir, f"_{os.path.splitext(os.path.basename(src))[0]}_inference.jpg")
        cv2.imwrite(inference_src, inference_img)
        print(f'  ✔ Prepared inference image at {inference_w}x{inference_h}')
        
        # Inference
        with torch.no_grad():
            prediction = predict_depth(model, inference_src, input_size=input_size)
        
        depth = prediction.depth
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        if depth.ndim == 3: depth = depth[0]

        # Resize model outputs back to the Unity-facing output size.
        depth = cv2.resize(depth, (output_w, output_h), interpolation=cv2.INTER_LINEAR)
        base = os.path.splitext(os.path.basename(src))[0]
        
        if save_16bit:
            # 16-bit raw depth (normalized by max available in frame)
            dmax = depth.max()
            if dmax > 0:
                depth16 = (depth / dmax * 65535).astype(np.uint16)
                cv2.imwrite(os.path.join(out_dir, f'{base}_depth_16bit.png'), depth16)
                
                # --- SAM2 Automatic Masks ---
                depth16_for_sam = cv2.resize(depth16, (inference_w, inference_h), interpolation=cv2.INTER_LINEAR)
                labels, num_regions = generate_sam2_label_map(inference_img, depth16_for_sam)
                labels = cv2.resize(
                    labels.astype(np.uint16),
                    (output_w, output_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)
                
                # Save Colored Segmentation
                labels_color = colorize_labels(labels, num_regions + 1)
                cv2.imwrite(os.path.join(out_dir, f'{base}_depth_segments.png'), labels_color)
                
                # --- Export raw mask data for downstream precise cropping ---
                cv2.imwrite(os.path.join(out_dir, f'{base}_depth_mask_raw.png'), labels.astype(np.uint8))
                np.save(os.path.join(out_dir, f'{base}_depth_mask.npy'), labels)
                
                print(f'  ✔ SAM2 Segmentation Saved (Found {num_regions} regions)')

    print(f"--- Done. Results saved to {out_dir} ---")

def main():
    IMAGE_PATH = "./assets/pictures/whisky.jpg"
    OUTDIR     = "./outputs/inference_results"
    MODEL_NAME = "depth-anything/da3-small"
    INPUT_SIZE = 768
    SAVE_16BIT = True
    TARGET_WIDTH = 1024
    INFERENCE_WIDTH = 768
    run_inference(
        IMAGE_PATH,
        OUTDIR,
        MODEL_NAME,
        INPUT_SIZE,
        SAVE_16BIT,
        TARGET_WIDTH,
        INFERENCE_WIDTH,
    )

if __name__ == '__main__':
    main()
