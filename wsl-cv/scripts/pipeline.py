import os
import argparse
from run_inference_depth_sam31 import run_inference
from cut_img import process_cut_img
from extract_track import extract_smooth_track
from level_designer import finalize_level, plan_track, repair_track_occlusions, surfaces_path_for
from muse_image import front_objects, load_setting, parse_styles, styles_from_env, stylize_run
from showcase import build_showcase

def run_pipeline(
    image_path: str,
    out_dir: str = "./outputs/inference_results",
    send2unity_dir: str = "./Send2Unity",
    styles: list[str] | None = None,
):
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
    
    # run_inference_depth_sam31 saves DA3 depth and the SAM 3.1 label map:
    # {base_name}_depth_16bit.png and {base_name}_depth_mask.npy
    depth16_path = os.path.join(out_dir, f"{base_name}_depth_16bit.png")
    mask_path = os.path.join(out_dir, f"{base_name}_depth_mask.npy")
    
    if not os.path.exists(depth16_path) or not os.path.exists(mask_path):
        print("Error: Step 1 failed to produce expected outputs.")
        return

    print(f"\n=============================================")
    print(f"--- Step 2: Choosing Track Surface (Muse Spark) ---")
    print(f"=============================================")
    objects_path = os.path.join(out_dir, f"{base_name}_sam31_objects.json")
    if os.path.exists(surfaces_path_for(mask_path)):
        os.remove(surfaces_path_for(mask_path))
    try:
        plan = plan_track(image_path, mask_path, objects_path)
    except Exception as exc:
        print(f"Muse Spark surface choice failed ({exc}); using the geometric track selection.")
        plan = None

    print(f"\n=============================================")
    print(f"--- Step 3: Cutting Image ---")
    print(f"=============================================")
    process_cut_img(image_path, depth16_path, mask_path, out_dir=send2unity_dir)
    
    # process_cut_img saves layer_00_mask.npy
    layer_00_mask_path = os.path.join(send2unity_dir, "layer_00_mask.npy")
    out_json = os.path.join(send2unity_dir, "track_points.json")
    
    if not os.path.exists(layer_00_mask_path):
        print("Error: Step 3 failed to produce layer_00_mask.npy.")
        return

    print(f"\n=============================================")
    print(f"--- Step 4: Extracting Track ---")
    print(f"=============================================")
    repair_track_occlusions(layer_00_mask_path, mask_path)
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
    finalize_level(image_path, mask_path, objects_path, send2unity_dir, plan)

    styles = styles_from_env() if styles is None else styles
    print(f"\n=============================================")
    print(f"--- Step 5: Background Fill + Art Styles (Muse Image: {', '.join(styles) or 'background only'}) ---")
    print(f"=============================================")
    try:
        stylize_run(
            image_path,
            send2unity_dir,
            styles,
            photo_background=True,
            setting=load_setting(objects_path),
            removed=front_objects(send2unity_dir, objects_path),
        )
    except Exception as exc:
        print(f"Muse Image failed ({exc}); the level keeps the photo's art and inpainted background.")

    print(f"\n=============================================")
    print(f"--- Step 6: Building 3D Showcase ---")
    print(f"=============================================")
    build_showcase(image_path, out_dir, send2unity_dir, os.path.join(send2unity_dir, "showcase.html"))

    print(f"\nPipeline completed successfully! Unity outputs are in {send2unity_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full SkateP CV pipeline on an image.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--out_dir", default="./outputs/inference_results", help="Directory for intermediate inference outputs")
    parser.add_argument("--send2unity_dir", default="./Send2Unity", help="Directory for final Unity outputs")
    parser.add_argument("--styles", default=None, help="Muse Image styles: comma-separated, 'all' or 'none' (default: $MUSE_IMAGE_STYLES or all)")
    args = parser.parse_args()
    
    run_pipeline(
        args.image_path,
        out_dir=args.out_dir,
        send2unity_dir=args.send2unity_dir,
        styles=parse_styles(args.styles) if args.styles is not None else None,
    )
