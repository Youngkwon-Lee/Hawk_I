"""Extract MediaPipe pose landmarks for PD4T gait 20-s clips (research QA, private outputs)."""
import os, glob, os, sys, json, time
from multiprocessing import Pool
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

from config import DATA_ROOT, CLIP_ROOT
MODEL = os.path.join(DATA_ROOT, "pose_landmarker_full.task")
OUT = os.path.join(DATA_ROOT, "npz")
TARGET_FPS = 15.0


def run(path):
    name = os.path.basename(path).replace("_clip.mp4", "")
    out = os.path.join(OUT, name + ".npz")
    if os.path.exists(out):
        return name, "skip"
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO, num_poses=2,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps / TARGET_FPS))
    t, img, world, ok, idx = [], [], [], [], 0
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                ts = idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(ts * 1000))
                t.append(ts)
                if res.pose_landmarks:
                    # choose the largest person (patient usually; assistants may appear)
                    areas = [ (max(p.x for p in pl)-min(p.x for p in pl))*(max(p.y for p in pl)-min(p.y for p in pl)) for pl in res.pose_landmarks]
                    k = int(np.argmax(areas))
                    img.append([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[k]])
                    world.append([[p.x, p.y, p.z] for p in res.pose_world_landmarks[k]])
                    ok.append(len(res.pose_landmarks))
                else:
                    img.append(np.full((33, 4), np.nan).tolist())
                    world.append(np.full((33, 3), np.nan).tolist())
                    ok.append(0)
            idx += 1
    cap.release()
    np.savez_compressed(out, t=np.array(t), img=np.array(img, dtype=np.float32),
                        world=np.array(world, dtype=np.float32), npose=np.array(ok, dtype=np.int8),
                        fps=fps, stride=stride)
    return name, f"{len(t)} frames, detected {int((np.array(ok)>0).mean()*100)}%"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CLIP_ROOT, "*", "*_clip.mp4")))
    t0 = time.time()
    with Pool(4) as pool:
        for i, (name, msg) in enumerate(pool.imap_unordered(run, files), 1):
            print(f"[{i}/{len(files)} {time.time()-t0:.0f}s] {name} {msg}", flush=True)
    print("DONE", flush=True)
