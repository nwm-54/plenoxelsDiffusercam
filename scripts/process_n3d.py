import os
import imageio.v3 as iio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import functools
import tqdm

def extract_video(video_path: Path, output_root: Path):
    print(video_path)
    cam_name = video_path.stem
    out_dir = output_root / cam_name / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(iio.imiter(video_path, plugin='pyav', thread_type="FRAME")):
        out_path = out_dir / f"{idx:04d}.png"
        if not out_path.exists():
            iio.imwrite(out_path, frame, plugin="pillow", compress_level=1)
            print(f"Extracted: {out_path}")

def main(source_path: Path, max_workers=4):
    videos = sorted(source_path.glob("*.mp4"))
    if not videos:
        print(f"No video files found in {source_path}.")
        return
    
    with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
        executor.map(functools.partial(extract_video, output_root=source_path), videos)

    print(f"Finished: {len(videos)} videos -> {source_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract frames from a video file.")
    p.add_argument("source", type=Path, help="Path to the video file.")
    p.add_argument("-w", "--workers", type=int, default=4, help="Number of worker threads.")
    args = p.parse_args()

    main(args.source.resolve(), args.workers)