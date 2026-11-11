"""Zero-shot far-end turn temporal grounding with Qwen3-VL (PD4T official test, 116 clips).

Protocol frozen before any test scoring (2026-11-11):
- model: Qwen3-VL-4B-Instruct, fp16, greedy decoding, max_new_tokens 256
- video: the 20 s clip, sampled at FPS frames/s, max_pixels MAXPIX per frame, native timestamps
- one fixed prompt (below); no prompt tuning on test; a format smoke test used a TRAIN video only
- output parsed as JSON list of {"start","end"} seconds; unparseable -> empty prediction (counted)
Run on the ICT GPU server (Quadro RTX 5000); score with eval_vlm.py.
"""
import os, sys, json, re, time, argparse
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL = os.environ.get("QWEN3VL_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
FPS = 4.0
MAXPIX = 320 * 320

PROMPT = (
    "This video shows a person with Parkinson's disease doing a clinical gait test: walk away from the "
    "camera, turn around at the far end, and walk back. The clip is {dur:.1f} seconds long.\n"
    "Task: find the far-end turnaround, i.e. the time when the person, at the end of the walkway far from "
    "the camera, rotates the body to reverse walking direction. Do NOT report turns close to the camera.\n"
    "Give the start and end time in seconds of each far-end turnaround (usually exactly one; zero if none "
    "is visible). Answer with JSON only, e.g. [{{\"start\": 7.2, \"end\": 8.4}}]."
)


def parse(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        out = []
        for a in arr:
            s, e = float(a["start"]), float(a["end"])
            if e < s:
                s, e = e, s
            out.append([s, e, (s + e) / 2])
        return out
    except Exception:
        return None


def patch_conv3d_as_linear(model):
    """The vision patch embedding is a Conv3d with kernel == stride, which is mathematically a linear
    layer on flattened patches. cuDNN Conv3d in fp16 on the Turing GPU stalled for minutes, so run the
    identical computation as a matmul. Numerically equivalent (no change to the model)."""
    pe = model.model.visual.patch_embed
    W = pe.proj.weight.reshape(pe.proj.weight.shape[0], -1)
    b = pe.proj.bias

    def fwd(hidden_states):
        x = hidden_states.reshape(-1, W.shape[1]).to(W.dtype)
        return torch.nn.functional.linear(x, W, b)
    pe.forward = fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", help="dir with <pid>/<visit>_<pid>_clip.mp4")
    ap.add_argument("--manifest", help="test manifest json (key, pid, visit, clip_dur)")
    ap.add_argument("--video", help="single video for a format smoke test")
    ap.add_argument("--dur", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    dev = f"cuda:{args.gpu}"
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16, device_map={"": dev},
                                                            attn_implementation="sdpa").eval()
    patch_conv3d_as_linear(model)
    if args.video:
        jobs = [{"key": "smoke", "path": args.video, "dur": args.dur}]
    else:
        man = json.load(open(args.manifest))
        jobs = [{"key": o["key"], "path": f"{args.clips}/{o['pid']}/{o['visit']}_{o['pid']}_clip.mp4",
                 "dur": o["clip_dur"]} for o in man]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["key"] for l in open(args.out)}
    fo = open(args.out, "a")
    for j in jobs:
        if j["key"] in done:
            continue
        t0 = time.time()
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": j["path"], "fps": FPS, "min_pixels": 64 * 32 * 32, "max_pixels": MAXPIX,
             "video_start": 0.0, "video_end": float(j["dur"])},
            {"type": "text", "text": PROMPT.format(dur=j["dur"])}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, videos, vk = process_vision_info(msgs, image_patch_size=16, return_video_kwargs=True,
                                                 return_video_metadata=True)
        metas = None
        if videos:
            videos, metas = map(list, zip(*videos))
        inp = proc(text=text, images=images, videos=videos, video_metadata=metas, return_tensors="pt",
                   do_resize=False, **vk).to(dev)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=256, do_sample=False)
        ans = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        segs = parse(ans)
        rec = {"key": j["key"], "raw": ans, "pred": segs if segs is not None else [], "parsed": segs is not None,
               "n_input_tokens": int(inp["input_ids"].shape[1]), "sec": round(time.time() - t0, 1)}
        fo.write(json.dumps(rec) + "\n"); fo.flush()
        print(json.dumps(rec)[:300], flush=True)
    meta = {"model": MODEL, "fps": FPS, "max_pixels": MAXPIX, "prompt": PROMPT, "dtype": "float16",
            "decoding": "greedy, max_new_tokens 256"}
    json.dump(meta, open(args.out + ".protocol.json", "w"), indent=1)


if __name__ == "__main__":
    main()
