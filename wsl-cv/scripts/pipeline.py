import os
import argparse
from run_inference_depth_sam2 import run_inference
from cut_img import process_cut_img
from extract_track import extract_smooth_track

def run_pipeline(image_path: str, out_dir: str = "./outputs/inference_results", send2unity_dir: str = "./Send2Unity"):
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(send2unity_dir, exist_ok=True)
    
    print(f"\n=============================================")
    print(f"--- Step 1: Running Inference for {image_path} ---")
    print(f"=============================================")
    run_inference(image_path, out_dir=out_dir)
    
    # run_inference_depth_sam2 saves DA3 depth and the SAM2 label map:
    # {base_name}_depth_16bit.png and {base_name}_depth_mask.npy
    depth16_path = os.path.join(out_dir, f"{base_name}_depth_16bit.png")
    mask_path = os.path.join(out_dir, f"{base_name}_depth_mask.npy")
    
    if not os.path.exists(depth16_path) or not os.path.exists(mask_path):
        print("Error: Step 1 failed to produce expected outputs.")
        return

    print(f"\n=============================================")
    print(f"--- Step 2: Cutting Image ---")
    print(f"=============================================")
    process_cut_img(image_path, depth16_path, mask_path, out_dir=send2unity_dir)
    
    # process_cut_img saves layer_00_mask.npy
    layer_00_mask_path = os.path.join(send2unity_dir, "layer_00_mask.npy")
    out_json = os.path.join(send2unity_dir, "track_points.json")
    
    if not os.path.exists(layer_00_mask_path):
        print("Error: Step 2 failed to produce layer_00_mask.npy.")
        return

    print(f"\n=============================================")
    print(f"--- Step 3: Extracting Track ---")
    print(f"=============================================")
    extract_smooth_track(
        layer_00_mask_path,
        out_json,
        rgb_image_path=image_path,
        source_img_name=os.path.basename(image_path),
        smooth_sigma=18.0,
        epsilon=18.0,
        alpha=0.75,
        sample_interval=18.0,
        clip_smooth_sigma=1.5,
        clip_top_margin_px=80.0,
    )
    
    print(f"\nPipeline completed successfully! Unity outputs are in {send2unity_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full SkateP CV pipeline on an image.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--out_dir", default="./outputs/inference_results", help="Directory for intermediate inference outputs")
    parser.add_argument("--send2unity_dir", default="./Send2Unity", help="Directory for final Unity outputs")
    args = parser.parse_args()
    
    run_pipeline(args.image_path, out_dir=args.out_dir, send2unity_dir=args.send2unity_dir)
