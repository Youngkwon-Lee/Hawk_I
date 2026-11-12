"""QLoRA fine-tuning of Qwen3-VL-4B for far-end turn temporal grounding (PD4T official TRAIN, 310 clips).

Recipe fixed in advance (no validation split, following the paper protocol; test is never touched here):
- 4-bit NF4 base (fp16 compute), vision tower and aligner frozen and not quantised
- LoRA r=8, alpha=32, dropout 0.05 on the language model's linear layers
- AdamW lr 1e-4, 10 warmup steps then cosine, batch 1 x grad-accum 2, 2 epochs, seed 0, per-epoch order seed 1000+epoch
- same video sampling as zero-shot (4 fps, max 320x320 px/frame); prompt v2 (one or two turns)
- target = clinician far-end turn intervals as JSON, seconds rounded to 0.1; loss on answer tokens only
Run on ICT (Quadro RTX 5000 16 GB, ~13 GB peak, ~33 s per optimizer step); see ft_loop.sh for auto-resume.
The 2026-11-12 run: 308 optimizer steps (4 train clips skipped on CUDA OOM), resumed twice (WSL restart,
CUDA allocator assert with expandable_segments).
"""
import os, sys, json, math, random, time, argparse
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from qwen_vl_utils import process_vision_info
from qwen_turn_zeroshot import MODEL, FPS, MAXPIX, PROMPTS, patch_conv3d_as_linear


def build(proc, path, dur, prompt, answer=None):
    msgs = [{"role": "user", "content": [
        {"type": "video", "video": path, "fps": FPS, "min_pixels": 64 * 32 * 32, "max_pixels": MAXPIX,
         "video_start": 0.0, "video_end": float(dur)},
        {"type": "text", "text": prompt.format(dur=dur)}]}]
    images, videos, vk = process_vision_info(msgs, image_patch_size=16, return_video_kwargs=True,
                                             return_video_metadata=True)
    videos, metas = map(list, zip(*videos))
    p_text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = p_text + answer + "<|im_end|>\n"
    enc = proc(text=full, images=images, videos=videos, video_metadata=metas, return_tensors="pt",
               do_resize=False, **vk)
    n_prompt = proc(text=p_text, images=images, videos=videos, video_metadata=metas, return_tensors="pt",
                    do_resize=False, **vk)["input_ids"].shape[1]
    labels = enc["input_ids"].clone(); labels[:, :n_prompt] = -100
    enc["labels"] = labels
    return enc


def answer_of(lab):
    return json.dumps([{"start": round(s, 1), "end": round(e, 1)} for s, e in lab])


def save_state(model, opt, sched, scaler, step, ep, i, path):
    torch.save({"lora": get_peft_model_state_dict(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "step": step, "ep": ep, "i": i}, path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=0, help="smoke test only")
    args = ap.parse_args()
    random.seed(0); torch.manual_seed(0)
    dev = f"cuda:{args.gpu}"
    proc = AutoProcessor.from_pretrained(MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16,
                             llm_int8_skip_modules=["visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16, quantization_config=bnb,
                                                            device_map={"": dev}, attn_implementation="sdpa")
    patch_conv3d_as_linear(model)
    _gvf = model.model.get_video_features

    def gvf_no_grad(*a, **k):  # frozen vision tower: keep it out of the autograd graph (saves ~1 GiB)
        with torch.no_grad():
            return _gvf(*a, **k)
    model.model.get_video_features = gvf_no_grad
    # not prepare_model_for_kbit_training: it upcasts the frozen fp16 vision tower / embeddings to fp32
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    man = json.load(open(args.manifest))
    items = [(f"{args.clips}/{o['pid']}/{o['visit']}_{o['pid']}_clip.mp4", o["clip_dur"], answer_of(o["lab"]))
             for o in man]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
    accum = 2
    total = args.max_steps or math.ceil(len(items) * args.epochs / accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / 10 if s < 10 else 0.5 * (1 + math.cos(math.pi * (s - 10) / max(1, total - 10))))
    scaler = torch.amp.GradScaler("cuda")
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    step = 0; t0 = time.time(); model.train()
    ck = os.path.join(args.out, "resume.pt"); start_ep, start_i = 0, 0
    if os.path.exists(ck):  # resume: same per-epoch order, skip finished samples
        st = torch.load(ck, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, st["lora"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        step, start_ep, start_i = st["step"], st["ep"], st["i"]
        log.write(json.dumps({"resumed_at_step": step, "ep": start_ep, "i": start_i}) + "\n"); log.flush()
    for ep in range(args.epochs):
        order = list(range(len(items))); random.Random(1000 + ep).shuffle(order)
        run = 0.0
        for i, k in enumerate(order):
            if ep < start_ep or (ep == start_ep and i < start_i):
                continue
            path, dur, ans = items[k]
            enc = build(proc, path, dur, PROMPTS["v2"], ans).to(dev)
            labels = enc.pop("labels")
            n_ans = int((labels[0] != -100).sum())
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**enc, logits_to_keep=n_ans + 1).logits  # answer span only
                    loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), labels[0, -n_ans:]) / accum
                scaler.scale(loss).backward(); run += loss.item()
            except RuntimeError as err:  # torch.OutOfMemoryError or "CUDA driver error: out of memory"
                if "out of memory" not in str(err):
                    raise
                logits = loss = None
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                log.write(json.dumps({"oom_skip": k, "n_tokens": int(enc["input_ids"].shape[1])}) + "\n"); log.flush()
                continue
            if (i + 1) % accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
                rec = {"step": step, "epoch": ep, "loss": run, "lr": sched.get_last_lr()[0],
                       "elapsed_s": round(time.time() - t0), "mem_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)}
                log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True); run = 0.0
                if step % 20 == 0:
                    save_state(model, opt, sched, scaler, step, ep, i + 1, ck)
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break
        model.save_pretrained(os.path.join(args.out, f"adapter_epoch{ep + 1}"))
        save_state(model, opt, sched, scaler, step, ep + 1, 0, ck)
    json.dump({"recipe": __doc__, "steps": step, "n_train": len(items), "base": MODEL},
              open(os.path.join(args.out, "recipe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
