"""Interactive recorder: guides a signer through the vocabulary, one clip at a time.

Keys
  READY   : SPACE start (3-2-1 countdown) | S skip item | U undo last saved clip | Q / ESC quit
  RECORD  : SPACE stop (auto-stops at the time limit)
  REVIEW  : ENTER / SPACE accept | R retake | S skip item
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path


from .capture import VideoSource, draw_landmarks, quality_issues
from .dataset import ISLDataset
from .landmarks import FrameLandmarks, HolisticExtractor, RawSequence
from .normalize import find_active_span
from .vocabulary import Sentence, Sign


@dataclass
class Task:
    clip_type: str            # "sign" | "sentence"
    item: Sign | Sentence
    rep: int                  # which repetition this will be (1-based)
    of: int                   # target repetitions


def build_tasks(ds: ISLDataset, signer_id: str, signs: list[Sign], sentences: list[Sentence],
                types: tuple[str, ...] = ("sign", "sentence"), reps_sign: int = 5, reps_sentence: int = 3,
                only: set[str] | None = None, shuffle: bool = True, seed: int = 0) -> list[Task]:
    """Everything this signer still has to record (resumable: existing OK clips are counted).

    Items are ordered in *rounds* (one repetition of every item, then the next repetition) so
    that fatigue and drift are not correlated with a particular sign."""
    rng = random.Random(f"{seed}-{signer_id}")
    tasks: list[Task] = []
    for clip_type, items, reps in (("sign", signs, reps_sign), ("sentence", sentences, reps_sentence)):
        if clip_type not in types:
            continue
        pool = [i for i in items if not only or i.id in only or getattr(i, "gloss", "") in only]
        need = {i.id: reps - ds.count(signer_id, clip_type, i.id) for i in pool}
        for rnd in range(max(need.values(), default=0)):
            batch = [i for i in pool if need[i.id] > rnd]
            if shuffle:
                rng.shuffle(batch)
            tasks += [Task(clip_type, i, reps - need[i.id] + rnd + 1, reps) for i in batch]
    return tasks


class Recorder:
    WINDOW = "Signvoice recorder"

    def __init__(self, ds: ISLDataset, signer_id: str, tasks: list[Task], source: int | str = 0,
                 width: int = 1280, height: int = 720, fps: int = 30, countdown: float = 3.0,
                 max_seconds: dict | None = None, save_video: bool = False, session: str = "",
                 model_complexity: int = 1, auto_trim: bool = True):
        signer = ds.signer(signer_id)
        if not signer or signer["consent"] != "yes":
            raise PermissionError(
                f"signer '{signer_id}' is not registered with recorded consent. Run:\n"
                f"  python scripts/record_dataset.py register --signer {signer_id} --dominant right --consent-confirmed")
        self.ds, self.signer_id, self.tasks = ds, signer_id, tasks
        self.source_args = (source, width, height, fps)
        self.countdown, self.save_video, self.session = countdown, save_video, session
        self.max_seconds = max_seconds or {"sign": 6.0, "sentence": 12.0}
        self.model_complexity, self.auto_trim = model_complexity, auto_trim
        self._pending = None

    # -- drawing -------------------------------------------------------------
    @staticmethod
    def _text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
        import cv2
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _overlay(self, img, task: Task, idx: int, state: str, extra: str, issues: list[str]):
        import cv2
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 92), (20, 20, 20), -1)
        label = task.item.gloss if task.clip_type == "sign" else task.item.text
        self._text(img, f"[{task.clip_type.upper()}] {label}", (14, 36), 1.0, (140, 255, 200), 2)
        sub = f"take {task.rep}/{task.of}   |   clip {idx + 1}/{len(self.tasks)}   |   signer {self.signer_id}"
        if task.clip_type == "sentence":
            sub = "signs: " + task.item.gloss + "   |   " + sub
        self._text(img, sub, (14, 72), 0.55, (210, 210, 210), 1)
        colors = {"ready": (120, 220, 120), "countdown": (0, 200, 255), "rec": (60, 60, 255), "review": (255, 200, 80)}
        hint = {"ready": "SPACE: start   S: skip   U: undo last   Q: quit",
                "countdown": "get ready - start signing at GO",
                "rec": "RECORDING - SPACE to stop",
                "review": "ENTER: accept   R: retake   S: skip"}[state]
        cv2.rectangle(img, (0, h - 44), (w, h), (20, 20, 20), -1)
        self._text(img, f"{state.upper()}  {extra}", (14, h - 16), 0.7, colors[state], 2)
        self._text(img, hint, (w // 2 - 120, h - 16), 0.55, (230, 230, 230), 1)
        if state == "rec":
            cv2.circle(img, (w - 36, 46), 12, (0, 0, 255), -1)
        for k, msg in enumerate(issues[:4]):
            self._text(img, "! " + msg, (14, 122 + 26 * k), 0.6, (80, 180, 255), 2)

    # -- main loop -----------------------------------------------------------
    def run(self) -> int:
        import cv2
        saved = 0
        source = VideoSource(*self.source_args)
        w, h = source.size
        aspect = w / h
        extractor = HolisticExtractor(model_complexity=self.model_complexity)
        idx, state = 0, "ready"
        frames: list[FrameLandmarks] = []
        stamps: list[float] = []
        writer, tmp_video = None, None
        t_go = t_start = 0.0
        pending: tuple[RawSequence, tuple[int, int], list[str]] | None = None
        history: list[tuple[int, str]] = []   # (task index, clip_id) of saved clips, for undo
        flash = ""

        def reset_take():
            nonlocal frames, stamps, writer, pending
            frames, stamps, pending = [], [], None
            if writer is not None:
                writer.release()
                writer = None
            if tmp_video and Path(tmp_video).exists():
                Path(tmp_video).unlink()

        try:
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
            while idx < len(self.tasks):
                frame, _ = source.read()
                if frame is None:
                    print("camera stopped delivering frames")
                    break
                now = time.monotonic()
                task = self.tasks[idx]
                fl = extractor.process(frame)  # un-mirrored frame -> correct left/right

                if state == "countdown" and now >= t_go:
                    state, t_start = "rec", now
                    if self.save_video:
                        vdir = self.ds.root / "videos"
                        vdir.mkdir(exist_ok=True)
                        tmp_video = str(vdir / "_current.mp4")
                        writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 source.nominal_fps or 30, (w, h))
                if state == "rec":
                    frames.append(fl)
                    stamps.append(now - t_start)
                    if writer is not None:
                        writer.write(frame)
                    if stamps[-1] >= self.max_seconds[task.clip_type]:
                        state = self._finish(frames, stamps, aspect, w, h)
                        pending = self._pending
                        if writer is not None:
                            writer.release(); writer = None

                shown = cv2.flip(frame, 1)  # selfie view for the signer; processing used the raw frame
                draw_landmarks(shown, fl, mirror=True)
                extra = ""
                if state == "countdown":
                    extra = f"{max(t_go - now, 0):.0f}" if t_go - now > 0.5 else "GO"
                elif state == "rec":
                    extra = f"{stamps[-1]:.1f}s  {len(frames)} frames"
                elif state == "review" and pending:
                    extra = f"{len(pending[0])} frames, active {pending[1][1] - pending[1][0]}"
                elif flash:
                    extra = flash
                self._overlay(shown, task, idx, state, extra, pending[2] if (state == "review" and pending) else [])
                cv2.imshow(self.WINDOW, shown)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                if state == "ready":
                    flash = ""
                    if key == ord(" "):
                        state, t_go = "countdown", now + self.countdown
                    elif key == ord("s"):
                        idx += 1
                    elif key == ord("u") and history:
                        idx, cid = history.pop()
                        self.ds.set_status(cid, "rejected")
                        saved -= 1
                        flash = f"undid {cid}"
                elif state == "rec" and key == ord(" "):
                    state = self._finish(frames, stamps, aspect, w, h)
                    pending = self._pending
                    if writer is not None:
                        writer.release(); writer = None
                elif state == "review":
                    if key in (13, ord(" ")):
                        raw, span, _ = pending
                        if writer is not None:
                            writer.release(); writer = None
                        row = self.ds.add_clip(raw, task.clip_type, task.item, self.signer_id, self.session,
                                               trim=span, video_src=tmp_video if self.save_video else None)
                        history.append((idx, row["clip_id"]))
                        saved += 1
                        flash = f"saved {row['clip_id']}"
                        reset_take(); idx += 1; state = "ready"
                    elif key == ord("r"):
                        reset_take(); state = "ready"
                    elif key == ord("s"):
                        reset_take(); idx += 1; state = "ready"
        finally:
            reset_take()
            extractor.close()
            source.release()
            cv2.destroyAllWindows()
        print(f"Saved {saved} clip(s) for {self.signer_id}. Tasks completed: {idx}/{len(self.tasks)}")
        return saved

    def _finish(self, frames, stamps, aspect, w, h) -> str:
        raw = RawSequence.from_frames(frames, stamps, aspect, meta={"width": w, "height": h, "session": self.session})
        span = find_active_span(raw) if self.auto_trim else (0, len(raw))
        self._pending = (raw, span, quality_issues(raw, span))
        return "review"
