"""
Lightgo icon generator
Outputs: resources/lightgo.ico  (16/32/48/64/128/256 multi-size)

Run once before cargo build:
    python icon_gen.py
"""

from PIL import Image, ImageDraw, ImageFilter
import os, math

def make_icon(size: int) -> Image.Image:
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx = cy = size / 2.0
    r  = size / 2.0 - size * 0.05        # outer radius
    pad = size * 0.05

    # ── 1. Dark circle background ────────────────────────────
    draw.ellipse([pad, pad, size - pad, size - pad],
                 fill=(13, 17, 23, 255))

    # ── 2. Amber glow halo (blurred overlay) ────────────────
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd   = ImageDraw.Draw(glow)
    gr   = r * 0.55
    gd.ellipse([cx - gr, cy - gr, cx + gr, cy + gr],
               fill=(251, 191, 36, 55))
    glow = glow.filter(ImageFilter.GaussianBlur(size * 0.14))
    img  = Image.alpha_composite(img, glow)
    draw = ImageDraw.Draw(img)

    # ── 3. Lightning bolt (normalized → scaled) ─────────────
    # Vertices as fractions of size, designed for a clean bolt
    nv = [
        (0.578, 0.078),   # top-right
        (0.234, 0.531),   # mid-left
        (0.422, 0.531),   # mid-notch
        (0.391, 0.938),   # bottom tip
        (0.734, 0.453),   # right
        (0.547, 0.453),   # right-notch
    ]
    bolt = [(x * size, y * size) for x, y in nv]
    draw.polygon(bolt, fill=(251, 191, 36, 255))

    # ── 4. Thin amber circle border ─────────────────────────
    lw = max(1, round(size * 0.032))
    draw.ellipse([pad, pad, size - pad, size - pad],
                 outline=(251, 191, 36, 190), width=lw)

    # ── 5. "LG" text — only visible at 128px+ ───────────────
    if size >= 128:
        try:
            from PIL import ImageFont
            fsize = max(10, size // 9)
            font  = ImageFont.truetype("arial.ttf", fsize)
        except Exception:
            font = ImageFont.load_default()
        text   = "LIGHTGO"
        bbox   = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = cx - tw / 2
        ty = size * 0.82
        draw.text((tx, ty), text, font=font,
                  fill=(251, 191, 36, 160))

    return img


def main():
    os.makedirs("resources", exist_ok=True)
    sizes = [16, 32, 48, 64, 128, 256]
    frames = [make_icon(s) for s in sizes]

    out = "resources/lightgo.ico"
    frames[0].save(
        out, format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=frames[1:],
    )
    print(f"Saved: {out}  ({', '.join(str(s) for s in sizes)} px)")

    # also save a PNG preview
    frames[-1].save("resources/lightgo_256.png")
    print("Preview: resources/lightgo_256.png")


if __name__ == "__main__":
    main()
