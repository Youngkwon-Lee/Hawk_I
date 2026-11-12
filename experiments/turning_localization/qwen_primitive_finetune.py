"""RGB + pose early fusion for the primitives pose alone measures poorly (Qwen3-VL-4B QLoRA, ICT).

Primitives (0-3): arm_swing_asymmetry, trunk_flexion, step_length_asymmetry, postural_instability.
Input  = the 20-s clip (8 frames, uniform, max 448x448 px; spatial detail matters more than timing here)
         + the clip's automatic pose measurements written as text in the prompt (early fusion).
Target = clinician ratings as JSON (a missing rating is trained as 0; affects <=7 clips per item).
Recipe (fixed before any test scoring): 4-bit NF4 base, frozen vision tower, LoRA r=8/alpha=32 on the language
model, AdamW 1e-4, warmup 10 + cosine, batch 1 x accum 2, 2 epochs, seed 0, per-epoch order seed 1000+epoch,
resumable (resume.pt every 20 steps). Same code path as qwen_turn_finetune.py.
Runs: --train-folds all (final model, predicts official test) and --train-folds 0 / 1 (2-fold subject split
inside train; each predicts the other half -> out-of-fold train predictions for the bottleneck integrator).

usage:
  python qwen_primitive_finetune.py train   --manifest M --clips DIR --out OUT --train-folds all|0|1 [--gpu G]
  python qwen_primitive_finetune.py predict --manifest M --clips DIR --adapter A --which test|fold0|fold1 --out F.jsonl
"""
import os, sys, json, math, random, time, re, argparse
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, PeftModel
from qwen_vl_utils import process_vision_info
from qwen_turn_zeroshot import MODEL, patch_conv3d_as_linear

PRIMS = ["arm_swing_asymmetry", "trunk_flexion", "step_length_asymmetry", "postural_instability"]
NFRAMES, MAXPIX = 8, 448 * 448

PROMPT = (
    "This {dur:.0f}-second video shows a person with Parkinson's disease doing a clinical gait test (walk away "
    "from the camera, turn, walk back).\n"
    "Automatic pose measurements for this clip: arm swing amplitude left {arm_swing_amp_left:.2f} and right "
    "{arm_swing_amp_right:.2f} (torso lengths), arm swing asymmetry ratio {arm_swing_asym_ratio:.2f}, median trunk "
    "lean {trunk_lean_deg_median:.1f} degrees, ankle gap variability {ankle_gap_std:.3f}, vertical hip bob "
    "{hip_vertical_bob:.4f}.\n"
    "Using both the video and the measurements, rate each gait feature from 0 (normal) to 3 (severe): "
    "arm_swing_asymmetry, trunk_flexion, step_length_asymmetry, postural_instability. "
    "Answer with JSON only, e.g. {{\"arm_swing_asymmetry\": 0, \"trunk_flexion\": 0, "
    "\"step_length_asymmetry\": 0, \"postural_instability\": 0}}."
)


def messages(o, clips):
    path = f"{clips}/{o['pid']}/{o['visit']}_{o['pid']}_clip.mp4"
    return [{"role": "user", "content": [
        {"type": "video", "video": path, "nframes": NFRAMES, "min_pixels": 64 * 32 * 32, "max_pixels": MAXPIX,
         "video_start": 0.0, "video_end": float(o["clip_dur"])},
        {"type": "text", "text": PROMPT.format(dur=o["clip_dur"], **o["pose"])}]}]


def encode(proc, msgs, answer=None):
    images, videos, vk = process_vision_info(msgs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    videos, metas = map(list, zip(*videos))
    p_text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    kw = dict(images=images, videos=videos, video_metadata=metas, return_tensors="pt", do_resize=False, **vk)
    if answer is None:
        return proc(text=p_text, **kw), None
    enc = proc(text=p_text + answer + "<|im_end|>\n", **kw)
    n_prompt = proc(text=p_text, **kw)["input_ids"].shape[1]
    return enc, enc["input_ids"].shape[1] - n_prompt


def answer_of(o):
    return json.dumps({k: int(o["primitives"][k] or 0) for k in PRIMS})


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0))
        return {k: int(max(0, min(3, round(float(d[k]))))) for k in PRIMS}
    except Exception:
        return None


def load_model(dev, train):
    proc = AutoProcessor.from_pretrained(MODEL)
    if train:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16, llm_int8_skip_modules=["visual", "lm_head"])
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16, quantization_config=bnb,
                                                                device_map={"": dev}, attn_implementation="sdpa")
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16, device_map={"": dev},
                                                                attn_implementation="sdpa").eval()
    patch_conv3d_as_linear(model)
    return proc, model


def train(args):
    random.seed(0); torch.manual_seed(0)
    dev = f"cuda:{args.gpu}"
    man = [o for o in json.load(open(args.manifest)) if o["split"] == "train"]
    if args.train_folds != "all":
        man = [o for o in man if o["fold2"] == int(args.train_folds)]
    proc, model = load_model(dev, True)
    gvf = model.model.get_video_features
    model.model.get_video_features = lambda *a, **k: (torch.no_grad()(gvf))(*a, **k)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                           target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
    accum, epochs = 2, 2
    total = math.ceil(len(man) * epochs / accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / 10 if s < 10 else 0.5 * (1 + math.cos(math.pi * (s - 10) / max(1, total - 10))))
    scaler = torch.amp.GradScaler("cuda")
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    ck = os.path.join(args.out, "resume.pt"); step, start_ep, start_i = 0, 0, 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, st["lora"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); step, start_ep, start_i = st["step"], st["ep"], st["i"]
        log.write(json.dumps({"resumed_at_step": step, "ep": start_ep, "i": start_i}) + "\n"); log.flush()
    save = lambda ep, i: (torch.save({"lora": get_peft_model_state_dict(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                                      "scaler": scaler.state_dict(), "step": step, "ep": ep, "i": i}, ck + ".tmp"), os.replace(ck + ".tmp", ck))
    t0 = time.time(); model.train()
    for ep in range(epochs):
        order = list(range(len(man))); random.Random(1000 + ep).shuffle(order); run = 0.0
        for i, k in enumerate(order):
            if ep < start_ep or (ep == start_ep and i < start_i):
                continue
            enc, n_ans = encode(proc, messages(man[k], args.clips), answer_of(man[k]))
            enc = enc.to(dev); ids = enc["input_ids"]
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**enc, logits_to_keep=n_ans + 1).logits
                    loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids[0, -n_ans:]) / accum
                scaler.scale(loss).backward(); run += loss.item()
            except RuntimeError as err:
                if "out of memory" not in str(err):
                    raise
                logits = loss = None; opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                log.write(json.dumps({"oom_skip": man[k]["key"]}) + "\n"); log.flush(); continue
            if (i + 1) % accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
                log.write(json.dumps({"step": step, "epoch": ep, "loss": run, "elapsed_s": round(time.time() - t0),
                                      "mem_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)}) + "\n"); log.flush(); run = 0.0
                if step % 20 == 0:
                    save(ep, i + 1)
        model.save_pretrained(os.path.join(args.out, f"adapter_epoch{ep + 1}")); save(ep + 1, 0)
    json.dump({"recipe": __doc__, "steps": step, "n_train": len(man), "train_folds": args.train_folds, "base": MODEL,
               "prompt": PROMPT, "nframes": NFRAMES, "max_pixels": MAXPIX}, open(os.path.join(args.out, "recipe.json"), "w"), indent=1)


def predict(args):
    dev = f"cuda:{args.gpu}"
    man = json.load(open(args.manifest))
    sel = {"test": lambda o: o["split"] == "test", "fold0": lambda o: o["split"] == "train" and o["fold2"] == 0,
           "fold1": lambda o: o["split"] == "train" and o["fold2"] == 1}[args.which]
    man = [o for o in man if sel(o)]
    proc, model = load_model(dev, False)
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    done = {json.loads(l)["key"] for l in open(args.out)} if os.path.exists(args.out) else set()
    fo = open(args.out, "a")
    for o in man:
        if o["key"] in done:
            continue
        enc, _ = encode(proc, messages(o, args.clips)); enc = enc.to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=64, do_sample=False)
        ans = proc.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        pr = parse(ans)
        fo.write(json.dumps({"key": o["key"], "raw": ans, "pred": pr, "parsed": pr is not None}) + "\n"); fo.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--manifest", required=True); ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--train-folds", default="all"); ap.add_argument("--adapter"); ap.add_argument("--which", default="test")
    a = ap.parse_args()
    train(a) if a.mode == "train" else predict(a)
