"""Contact sheets: clinician label (yellow bar), predicted far turn (red), predicted near turn (green)."""
import os, json, random, subprocess, os, sys
from PIL import Image, ImageDraw, ImageFont
from config import DATA_ROOT, CLIP_ROOT

rows = json.load(open(os.path.join(DATA_ROOT, "turn_candidates.json")))
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
train = [r for r in rows if r["split"] == "train"]
pick = random.sample(train, 4)
font = ImageFont.load_default()
step = 0.5
os.makedirs("viz", exist_ok=True)
for i, r in enumerate(pick):
    vid, pid = r["clip"].rsplit("_", 1)
    p = os.path.join(CLIP_ROOT, pid, f"{vid}_{pid}_clip.mp4")
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p]).decode())
    n = int(dur / step)
    tiles = []
    for k in range(n):
        out = f"viz/_t{k:03d}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{k*step:.2f}", "-i", p, "-frames:v", "1", "-vf", "scale=150:-1", out], check=True)
        tiles.append(Image.open(out).copy())
    w, h = tiles[0].size
    cols = 10
    nrow = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, nrow * (h + 22) + 30), "white")
    d = ImageDraw.Draw(sheet)
    preds = r["pred"]
    d.text((4, 4), f"#{i} {r['clip']} score={r['score']} LABEL(yellow)={[(x['start'], x['end']) for x in r['labels']]} PRED far(red)/near(green)={[(p_['kind'][0], round(p_['start'],1), round(p_['end'],1)) for p_ in preds]}", fill="black", font=font)
    for k, im in enumerate(tiles):
        t = k * step
        x = (k % cols) * w
        y = 30 + (k // cols) * (h + 22)
        sheet.paste(im, (x, y))
        for p_ in preds:
            if p_["start"] <= t <= p_["end"]:
                d.rectangle([x, y, x + w - 1, y + h - 1], outline="red" if p_["kind"] == "far" else "green", width=4)
        if any(L["start"] - 0.25 <= t <= L["end"] + 0.25 for L in r["labels"]):
            d.rectangle([x, y + h + 1, x + w - 1, y + h + 7], fill="gold")
        d.text((x + 3, y + h + 9), f"{t:.1f}s", fill="black", font=font)
    sheet.save(f"viz/sheet_{i}.jpg", quality=80)
    print(i, r["clip"], r["score"], r["labels"], preds)
