import argparse, glob, os, random
from PIL import Image

def collect_images(folder, exts=(".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*{e}")))
    return sorted(paths)

def make_grid(input_dir: str, output: str, size: int = 32, tile: int = 16, seed: int | None = 42):
    n = tile * tile  # 16*16 = 256
    img_paths = collect_images(input_dir)
    if len(img_paths) < n:
        raise ValueError(f"Need at least {n} images, found {len(img_paths)} in {input_dir}")

    if seed is not None:
        random.seed(seed)
    picks = random.sample(img_paths, n)

    canvas = Image.new("RGB", (tile * size, tile * size), color=(0, 0, 0))
    for idx, p in enumerate(picks):
        im = Image.open(p)
        # force to RGB so modes match (e.g., grayscale/PNG with alpha)
        if im.mode != "RGB":
            im = im.convert("RGB")
        if im.size != (size, size):
            # nearest keeps pixel art sharp; switch to Image.BILINEAR for photos
            im = im.resize((size, size), Image.NEAREST)

        x = (idx % tile) * size
        y = (idx // tile) * size
        canvas.paste(im, (x, y))

    canvas.save(output)
    print(f"Saved grid: {output} ({tile}x{tile}, each {size}x{size})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create a 16x16 grid image from a folder of images.")
    ap.add_argument("input_dir", help="Folder containing images (at least 256).")
    ap.add_argument("-o", "--output", default="grid.png", help="Output image path.")
    ap.add_argument("--size", type=int, default=32, help="Cell size in pixels (default: 32).")
    ap.add_argument("--tile", type=int, default=16, help="Tiles per row/col (default: 16).")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (set None for non-deterministic).")
    args = ap.parse_args()
    make_grid(args.input_dir, args.output, size=args.size, tile=args.tile, seed=args.seed)