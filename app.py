"""
Vehicle Zone Dwell-Time Counter - 4 CIRCLE ZONES + PRO DASHBOARD  (YOLO11 + BoT-SORT, video file)
-------------------------------------------------------------------------------------------------
- Input : U:\\petrolpump\\petrol-pump.mp4  ->  Output: U:\\petrolpump\\petrol-pump_output.mp4
- 4 zones = 4 dots (circle). Gari ke box ke andar dot aaye to wo us zone me count hoti hai.
- Dashboard (right panel): unique vehicles, total visits, zones busy, har zone ka status + live timer,
  visits, avg time, last visit, recent visits.
- Log: dwell_log.csv | Stats: stats_YYYY-MM-DD.json | Zones: zones.json

Install:   pip install ultralytics opencv-python numpy
Run:       python app_video_4zones.py
Options:   --video --out --setup --edit --reset --scale 0.6 --stride 2 --no-show --start "2025-01-01 10:00:00"
Keys:      q quit | p pause | e dots edit | r dots dobara | x SARA data clear | [ ] threshold | + - w f 1 = window size
"""

import argparse
import csv
import glob
import json
import math
import os
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------- CONFIG (yahan tweak karo) -------------------------
VIDEO_PATH = r"U:\petrolpump\petrol-pump.mp4"
OUTPUT_PATH = None                  # None = "<video>_output.mp4"
NUM_ZONES = 4
CONF = 0.65
MIN_BOX_FRAC = 0.03
VEHICLE_CLASSES = [2, 3, 5, 7]      # COCO: car, motorcycle, bus, truck
ENTER_CONFIRM_SEC = 1.5
EXIT_GRACE_SEC = 3.0
BIG_VEHICLE_ZONE = 0                # 0 = band. Agar kisi zone (1-4) me bari gari ko priority deni ho to uska number likho
BOX_SHRINK = 0.10
REID_WINDOW_SEC = 6.0
LOST_KEEP_SEC = 30.0
STATIONARY_SPEED = 0.25
DET_CONF = 0.20
REID_IOU = 0.30
EDIT_ZONES_ON_START = False
RESET_ON_START = False
ZONES_FILE = "zones.json"
LOG_FILE = "dwell_log.csv"
LOG_HEADER = ["date", "vehicle_id", "type", "zone", "start", "end", "duration_sec", "duration"]
FONT = cv2.FONT_HERSHEY_SIMPLEX
DUPLEX = cv2.FONT_HERSHEY_DUPLEX

# colors (BGR)
BG = (33, 28, 24)
HEADER = (48, 40, 34)
CARD = (52, 45, 40)
LINE = (78, 70, 64)
TEXT = (245, 245, 245)
MUTED = (175, 165, 155)
RED = (70, 70, 240)
GREEN = (95, 200, 90)
AMBER = (0, 190, 255)
ZONE_COLORS = [(255, 170, 50), (60, 200, 255), (110, 220, 110), (230, 110, 220)]
# -----------------------------------------------------------------------------


def fmt(sec):
    sec = int(max(0, sec))
    return f"{sec // 3600:d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def ui_scale(frame):
    return max(1.0, frame.shape[1] / 1280.0)


def put(img, text, org, k, color=(255, 255, 255), size=0.6, thick=1):
    cv2.putText(img, text, org, FONT, size * k, color, max(1, int(round(thick * k))), cv2.LINE_AA)


def tag(img, text, x, y, k, bg, fg=(255, 255, 255), size=0.5):
    """Text ek rang ke background chip par."""
    sc, th = size * k, max(1, int(round(k)))
    (w, h), bl = cv2.getTextSize(text, FONT, sc, th)
    x = int(max(0, min(x, img.shape[1] - w - 10)))
    y = int(max(h + 8, y))
    cv2.rectangle(img, (x, y - h - 6), (x + w + 8, y + bl), bg, -1)
    cv2.putText(img, text, (x + 4, y - 2), FONT, sc, fg, th, cv2.LINE_AA)


def rrect(img, x1, y1, x2, y2, color, r=8):
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


# ------------------------------ WINDOW HELPERS ------------------------------
def screen_size():
    try:
        import tkinter as tk
        r = tk.Tk()
        r.withdraw()
        w, h = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        return w, h
    except Exception:
        return 1366, 768


class Viewer:
    def __init__(self, name, W, H, scale=None):
        self.name, self.W, self.H = name, W, H
        self.sw, self.sh = screen_size()
        self.scale = scale if scale else min(1.0, self.fit_scale(0.92, 0.80))
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Size %", name, int(self.scale * 100), 200, self._on_bar)

    def fit_scale(self, fw, fh):
        return min(self.sw * fw / self.W, self.sh * fh / self.H)

    def _on_bar(self, v):
        self.scale = max(0.10, v / 100.0)

    def set_scale(self, s):
        self.scale = max(0.10, min(2.0, s))
        cv2.setTrackbarPos("Size %", self.name, int(self.scale * 100))

    def show(self, img):
        s = self.scale
        if abs(s - 1.0) > 0.01:
            img = cv2.resize(img, (max(1, int(self.W * s)), max(1, int(self.H * s))),
                             interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        cv2.imshow(self.name, img)

    def to_frame(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def handle_key(self, key):
        if key in (ord("+"), ord("=")):
            self.set_scale(self.scale * 1.1)
        elif key == ord("-"):
            self.set_scale(self.scale * 0.9)
        elif key == ord("w"):
            self.set_scale(self.fit_scale(0.92, 0.80))
        elif key == ord("f"):
            self.set_scale(self.fit_scale(1.0, 0.93))
        elif key == ord("1"):
            self.set_scale(1.0)
        else:
            return False
        return True

    def close(self):
        cv2.destroyWindow(self.name)


# ------------------------------ VIDEO READER --------------------------------
class VideoReader:
    def __init__(self, path):
        if not os.path.exists(path):
            raise SystemExit(f"[!] Video file nahi mili: {path}")
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"[!] Video open nahi hui: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and not math.isnan(fps) and fps > 0 else 25.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pos = 0

    def read(self, stride=1):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        idx = self.pos
        self.pos += 1
        for _ in range(max(0, stride - 1)):
            if self.cap.grab():
                self.pos += 1
        return frame, idx

    def first_frame(self):
        ok, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.pos = 0
        if not ok:
            raise SystemExit("[!] Video ka pehla frame nahi parha.")
        return frame

    def release(self):
        self.cap.release()


# ------------------------------ ZONE SETUP (DOTS) ---------------------------
def draw_zones(frame, scale=None, existing=None):
    """4 dots lagao. Drag karke position badlo. u = undo | c = clear | ENTER = done"""
    H, W = frame.shape[:2]
    k = ui_scale(frame)
    win = f"SETUP - {NUM_ZONES} dots" + (" (edit)" if existing else "")
    pts = [list(p) for p in existing] if existing else []
    drag = {"i": None}
    grab_r = 28 * k

    def on_mouse(event, x, y, flags, param):
        fx, fy = view.to_frame(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            near = [(((p[0] - fx) ** 2 + (p[1] - fy) ** 2) ** 0.5, i) for i, p in enumerate(pts)]
            near = [n for n in near if n[0] <= grab_r]
            if near:
                drag["i"] = min(near)[1]
                pts[drag["i"]] = [fx, fy]
            elif len(pts) < NUM_ZONES:
                pts.append([fx, fy])
        elif event == cv2.EVENT_MOUSEMOVE and drag["i"] is not None:
            pts[drag["i"]] = [max(0, min(W - 1, fx)), max(0, min(H - 1, fy))]
        elif event == cv2.EVENT_LBUTTONUP:
            drag["i"] = None

    view = Viewer(win, W, H, scale)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = frame.copy()
        for i, p in enumerate(pts):
            c = ZONE_COLORS[i % 4]
            cv2.circle(img, tuple(p), int(14 * k), c, max(2, int(3 * k)))
            cv2.circle(img, tuple(p), int(4 * k), (0, 255, 255), -1)
            put(img, f"Z{i + 1}", (p[0] + int(18 * k), p[1] - int(10 * k)), k, c, 0.8, 2)
        if len(pts) < NUM_ZONES:
            msg = f"Dot {len(pts) + 1}/{NUM_ZONES} lagao | dot ko drag karke hilao | u=undo c=clear | +/- w f = size"
        else:
            msg = ("EDIT: dot drag karke hilao | ENTER = save (purana record delete hoga) | q = cancel"
                   if existing else f"{NUM_ZONES} dots ready - drag karke position badlo | ENTER = save")
        cv2.rectangle(img, (0, 0), (W, int(34 * k)), (0, 0, 0), -1)
        put(img, msg, (10, int(24 * k)), k, (255, 255, 255), 0.6, 2)
        view.show(img)
        key = cv2.waitKey(20) & 0xFF
        if view.handle_key(key):
            pass
        elif key == ord("u") and pts:
            pts.pop()
        elif key == ord("c"):
            pts.clear()
        elif key == 13 and len(pts) == NUM_ZONES:
            break
        elif key in (ord("q"), 27):
            view.close()
            if existing:
                return None
            raise SystemExit("Setup cancelled.")
    view.close()
    with open(ZONES_FILE, "w") as f:
        json.dump({"size": [W, H], "points": pts}, f)
    return pts


def load_zones(W, H):
    if not os.path.exists(ZONES_FILE):
        return None
    try:
        with open(ZONES_FILE) as f:
            d = json.load(f)
        pts = d["points"]
        assert len(pts) == NUM_ZONES and all(len(p) == 2 for p in pts)
        sw, sh = d.get("size", [W, H])
    except Exception:
        os.remove(ZONES_FILE)
        print(f"[i] Purani zones.json delete kar di - ab {NUM_ZONES} dots lagao")
        return None
    if (sw, sh) != (W, H):
        fx, fy = W / sw, H / sh
        pts = [[int(x * fx), int(y * fy)] for x, y in pts]
    return pts


class Zone:
    def __init__(self, idx, pt):
        self.idx = idx
        self.pt = (int(pt[0]), int(pt[1]))
        self.color = ZONE_COLORS[idx % len(ZONE_COLORS)]


def assign_dots(dets, zones, current=None):
    current = current or {}
    big = (BIG_VEHICLE_ZONE - 1) if BIG_VEHICLE_ZONE else None
    cand = {}
    for vid, cname, box, _cf in dets:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        x1, x2 = x1 + w * BOX_SHRINK, x2 - w * BOX_SHRINK
        y1, y2 = y1 + h * BOX_SHRINK, y2 - h * BOX_SHRINK
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        covered = {}
        for z in zones:
            px, py = z.pt
            if x1 <= px <= x2 and y1 <= py <= y2:
                covered[z.idx] = (px - cx) ** 2 + (py - cy) ** 2
        if not covered:
            continue
        if big is not None and len(covered) >= 2 and big in covered:
            zi = big
        elif current.get(vid) in covered:
            zi = current[vid]
        else:
            zi = min(covered, key=covered.get)
        d = covered[zi]
        if zi not in cand or d < cand[zi][0]:
            cand[zi] = (d, vid, cname)
    return {vid: (zi, cname) for zi, (d, vid, cname) in cand.items()}


# --------------------------- UNIQUE VEHICLE IDs -----------------------------
def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class IdManager:
    def __init__(self, next_id=1):
        self.next_id = next_id
        self.raw2vid = {}
        self.last = {}
        self.speed = {}

    def _stationary(self, vid):
        return self.speed.get(vid, 1.0) < STATIONARY_SPEED

    def hold_sec(self, vid):
        return LOST_KEEP_SEC if self._stationary(vid) else EXIT_GRACE_SEC

    def _relink_window(self, vid):
        return LOST_KEEP_SEC if self._stationary(vid) else REID_WINDOW_SEC

    @staticmethod
    def _match(new, old):
        i = iou(new, old)
        if i >= REID_IOU:
            return i
        ncx, ncy = (new[0] + new[2]) / 2, (new[1] + new[3]) / 2
        ocx, ocy = (old[0] + old[2]) / 2, (old[1] + old[3]) / 2
        inside = (old[0] <= ncx <= old[2] and old[1] <= ncy <= old[3]) or \
                 (new[0] <= ocx <= new[2] and new[1] <= ocy <= new[3])
        return i if (inside and i >= 0.12) else 0.0

    def assign(self, raw_id, box, now, taken):
        if raw_id in self.raw2vid:
            vid = self.raw2vid[raw_id]
        else:
            best, best_score = None, 0.0
            for v, (b, t) in self.last.items():
                if v in taken or now - t > self._relink_window(v):
                    continue
                sc = self._match(box, b)
                if sc > best_score:
                    best, best_score = v, sc
            if best is not None:
                vid = best
            else:
                vid = self.next_id
                self.next_id += 1
            self.raw2vid[raw_id] = vid
        prev = self.last.get(vid)
        if prev is not None:
            pb, pt = prev
            dt = now - pt
            if 0 < dt <= 1.0:
                w = max(1.0, box[2] - box[0])
                dx = (box[0] + box[2] - pb[0] - pb[2]) / 2
                dy = (box[1] + box[3] - pb[1] - pb[3]) / 2
                sp = (dx * dx + dy * dy) ** 0.5 / dt / w
                self.speed[vid] = 0.7 * self.speed.get(vid, 1.0) + 0.3 * sp
        self.last[vid] = (box, now)
        return vid

    def prune(self, now):
        for v in [v for v, (_, t) in self.last.items() if now - t > max(120, 2 * LOST_KEEP_SEC)]:
            del self.last[v]
            self.speed.pop(v, None)


# ------------------------------ VISIT TRACKING ------------------------------
class VisitTracker:
    def __init__(self):
        self.s = {}

    def occupant(self, zi):
        """Zone zi me abhi jo gari ACTIVE hai uski vid (ya None)."""
        best = None
        for v, st in self.s.items():
            if st["active"] and st["zone"] == zi and (best is None or st["start"] > self.s[best]["start"]):
                best = v
        return best

    def pending(self, zi):
        return [(v, st) for v, st in self.s.items() if st["zone"] == zi and not st["active"]]

    def busy_count(self):
        return sum(1 for zi in range(NUM_ZONES) if self.occupant(zi) is not None)

    def step(self, in_zone, now, hold=None):
        hold = hold or (lambda v: EXIT_GRACE_SEC)
        ev = []
        for vid, (zi, cls) in in_zone.items():
            st = self.s.get(vid)
            if st and st["zone"] != zi:
                if st["active"]:
                    ev.append(("end", vid, st["zone"], st["cls"], (st["start"], st["last"])))
                st = None
            if st is None:
                st = self.s[vid] = dict(zone=zi, first=now, last=now, active=False, start=None, cls=cls)
            st["last"], st["cls"] = now, cls
            if not st["active"] and now - st["first"] >= ENTER_CONFIRM_SEC:
                st["active"], st["start"] = True, st["first"]
                ev.append(("start", vid, zi, cls, st["start"]))
                for v, o in list(self.s.items()):
                    if v != vid and o["zone"] == zi and v not in in_zone:
                        if o["active"]:
                            ev.append(("end", v, zi, o["cls"], (o["start"], o["last"])))
                        del self.s[v]
        for vid, st in list(self.s.items()):
            limit = hold(vid) if st["active"] else EXIT_GRACE_SEC
            if vid not in in_zone and now - st["last"] >= limit:
                if st["active"]:
                    ev.append(("end", vid, st["zone"], st["cls"], (st["start"], st["last"])))
                del self.s[vid]
        return ev

    def close_all(self):
        ev = []
        for vid, st in self.s.items():
            if st["active"]:
                ev.append(("end", vid, st["zone"], st["cls"], (st["start"], st["last"])))
        self.s.clear()
        return ev


# ------------------------------- STATS --------------------------------------
class Stats:
    def __init__(self, day):
        self.day = day
        self.recent = []                      # (end_time_str, vid, cls, zone_idx, dur)
        self.last_dur = [None] * NUM_ZONES
        self.data = self._load()
        self._init_log()

    def _init_log(self):
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, newline="") as f:
                header = next(csv.reader(f), None)
            if header == LOG_HEADER:
                return
            os.replace(LOG_FILE, f"dwell_log_old_{int(time.time())}.csv")
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(LOG_HEADER)

    @staticmethod
    def _blank():
        return {"counts": [0] * NUM_ZONES, "dwell_sec": [0.0] * NUM_ZONES,
                "done": [0] * NUM_ZONES, "vehicle_ids": [], "next_id": 1}

    def _path(self):
        return f"stats_{self.day}.json"

    def _load(self):
        d = self._blank()
        if os.path.exists(self._path()):
            try:
                with open(self._path()) as f:
                    old = json.load(f)
                if len(old.get("counts", [])) == NUM_ZONES:       # purani (5 zone) file ignore
                    d.update(old)
            except Exception:
                pass
        self.vset = set(d["vehicle_ids"])
        return d

    def save(self):
        self.data["vehicle_ids"] = sorted(self.vset)
        with open(self._path(), "w") as f:
            json.dump(self.data, f, indent=2)

    def reset(self, keep_zones=False):
        for f in ([] if keep_zones else [ZONES_FILE]) + glob.glob("stats_*.json") + glob.glob("dwell_log*.csv"):
            if os.path.exists(f):
                os.remove(f)
        self.data = self._blank()
        self.vset = set()
        self.recent = []
        self.last_dur = [None] * NUM_ZONES
        self._init_log()
        print("[i] Sara data clear ho gaya.")

    def visit_started(self, zi, vid):
        self.data["counts"][zi] += 1
        self.vset.add(vid)

    def visit_ended(self, vid, cls, zi, s, e):
        dur = max(0.0, e - s)
        self.data["dwell_sec"][zi] += dur
        self.data["done"][zi] += 1
        end_str = datetime.fromtimestamp(e).strftime("%H:%M:%S")
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([self.day, f"V{vid}", cls, f"Z{zi + 1}",
                                    datetime.fromtimestamp(s).strftime("%H:%M:%S"), end_str,
                                    round(dur, 1), fmt(dur)])
        self.recent.append((end_str, vid, cls, zi, dur))
        self.recent = self.recent[-10:]
        self.last_dur[zi] = dur
        self.save()
        print(f"[Z{zi + 1}] V{vid} ({cls}) left | dwell = {fmt(dur)}")

    @property
    def unique(self):
        return len(self.vset)

    @property
    def visits(self):
        return sum(self.data["counts"])

    def avg(self, zi):
        d = self.data["done"][zi]
        return self.data["dwell_sec"][zi] / d if d else 0.0


def handle_events(events, stats):
    for kind, vid, zi, cls, data in events:
        if kind == "start":
            stats.visit_started(zi, vid)
            print(f"[Z{zi + 1}] V{vid} ({cls}) entered | unique = {stats.unique}")
        else:
            stats.visit_ended(vid, cls, zi, *data)


# ------------------------------ DRAWING -------------------------------------
def dashed_rect(img, p1, p2, col, th, dash=12):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), col, th)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), col, th)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), col, th)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), col, th)


def draw_zone_circles(frame, zones, tracker, now, k):
    """Circle zones: free = zone ka rang, detecting = amber, occupied = laal (bara circle) + label."""
    layer = frame.copy()
    info = {}
    for z in zones:
        ov = tracker.occupant(z.idx)
        pend = tracker.pending(z.idx)
        if ov is not None:
            col, r = RED, int(24 * k)
        elif pend:
            col, r = AMBER, int(18 * k)
        else:
            col, r = z.color, int(14 * k)
        info[z.idx] = (col, r, ov)
        cv2.circle(layer, z.pt, r, col, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.35, frame, 0.65, 0, frame)
    for z in zones:
        col, r, ov = info[z.idx]
        cv2.circle(frame, z.pt, r, col, max(2, int(3 * k)), cv2.LINE_AA)
        cv2.circle(frame, z.pt, int(4 * k), (0, 255, 255), -1, cv2.LINE_AA)
        label = f"ZONE {z.idx + 1}"
        if ov is not None:
            st = tracker.s[ov]
            label += f" | V{ov} {st['cls']} {fmt(now - st['start'])}"
        tag(frame, label, z.pt[0] + r + int(6 * k), z.pt[1] - int(4 * k), k, col,
            (255, 255, 255) if col == RED else (0, 0, 0), 0.5)


def panel_w(H):
    return int(430 * H / 860.0)


def build_panel(H, stats, tracker, zones, now, vsec, frac, conf):
    """Right-side professional dashboard (860 unit tall layout, H ke hisab se scale)."""
    s = H / 860.0
    PW = panel_w(H)
    p = np.full((H, PW, 3), BG, np.uint8)

    def u(v):
        return int(round(v * s))

    def txt(text, x, y, size=0.5, col=TEXT, th=1, font=FONT, right=False, center=False):
        sc, t = size * s, max(1, int(round(th * s)))
        xp = u(x)
        if right or center:
            w = cv2.getTextSize(text, font, sc, t)[0][0]
            xp = xp - w if right else xp - w // 2
        cv2.putText(p, text, (xp, u(y)), font, sc, col, t, cv2.LINE_AA)

    def box(x1, y1, x2, y2, col, r=8):
        rrect(p, u(x1), u(y1), u(x2), u(y2), col, u(r))

    # header + video progress
    cv2.rectangle(p, (0, 0), (PW, u(64)), HEADER, -1)
    txt("VEHICLE DWELL MONITOR", 16, 28, 0.62, TEXT, 1, DUPLEX)
    txt(f"{datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')}   |   video {fmt(vsec)}", 16, 51, 0.42, MUTED)
    cv2.rectangle(p, (0, u(64)), (PW, u(68)), LINE, -1)
    cv2.rectangle(p, (0, u(64)), (int(PW * max(0.0, min(1.0, frac))), u(68)), AMBER, -1)

    # KPI cards
    busy = tracker.busy_count()
    kpis = [("UNIQUE VEHICLES", str(stats.unique), TEXT),
            ("TOTAL VISITS", str(stats.visits), TEXT),
            ("ZONES BUSY", f"{busy}/{NUM_ZONES}", RED if busy else GREEN)]
    for j, (lab, val, col) in enumerate(kpis):
        x = 12 + j * 138
        box(x, 84, x + 130, 164, CARD)
        txt(lab, x + 10, 106, 0.33, MUTED)
        txt(val, x + 10, 148, 0.95, col, 2, DUPLEX)

    txt("ZONE STATUS", 14, 190, 0.42, MUTED)
    cv2.line(p, (u(110), u(186)), (PW - u(12), u(186)), LINE, 1)

    # zone cards
    for i, z in enumerate(zones):
        y = 202 + i * 126
        ov = tracker.occupant(i)
        pend = tracker.pending(i)
        if ov is not None:
            chip, chip_col, chip_fg = "OCCUPIED", RED, (255, 255, 255)
        elif pend:
            chip, chip_col, chip_fg = "DETECTING", AMBER, (0, 0, 0)
        else:
            chip, chip_col, chip_fg = "FREE", GREEN, (0, 0, 0)
        box(12, y, 418, y + 118, CARD)
        cv2.rectangle(p, (u(12), u(y + 6)), (u(18), u(y + 112)), z.color, -1)
        txt(f"ZONE {i + 1}", 32, y + 30, 0.62, TEXT, 1, DUPLEX)
        box(312, y + 12, 406, y + 36, chip_col, 12)
        txt(chip, 359, y + 29, 0.4, chip_fg, 1, FONT, center=True)

        if chip == "OCCUPIED":
            st = tracker.s[ov]
            txt(f"V{ov}  {st['cls'].upper()}", 32, y + 70, 0.72, TEXT, 1, DUPLEX)
            txt(fmt(now - st["start"]), 406, y + 70, 0.85, AMBER, 1, DUPLEX, right=True)
        elif chip == "DETECTING":
            v, pp = pend[0]
            txt(f"V{v}  {pp['cls'].upper()}  - confirming...", 32, y + 70, 0.55, MUTED)
        else:
            txt("No vehicle", 32, y + 70, 0.6, MUTED)
        cv2.line(p, (u(32), u(y + 80)), (u(406), u(y + 80)), LINE, 1)

        last = stats.last_dur[i]
        cols = [("VISITS", str(stats.data["counts"][i])),
                ("AVG TIME", fmt(stats.avg(i))),
                ("LAST VISIT", fmt(last) if last is not None else "--")]
        for j, (lab, val) in enumerate(cols):
            x = 32 + j * 124
            txt(lab, x, y + 96, 0.33, MUTED)
            txt(val, x, y + 114, 0.6, TEXT, 1, DUPLEX)

    # recent visits
    txt("RECENT VISITS", 14, 724, 0.42, MUTED)
    cv2.line(p, (u(120), u(720)), (PW - u(12), u(720)), LINE, 1)
    rows = list(reversed(stats.recent))[:4]
    if not rows:
        txt("(abhi koi visit complete nahi hui)", 16, 750, 0.42, MUTED)
    for n, (end_str, vid, cls, zi, dur) in enumerate(rows):
        y = 748 + n * 26
        cv2.circle(p, (u(20), u(y - 4)), u(5), ZONE_COLORS[zi % 4], -1, cv2.LINE_AA)
        txt(f"{end_str}   V{vid} {cls}   Z{zi + 1}", 32, y, 0.42, TEXT)
        txt(fmt(dur), 412, y, 0.45, AMBER, 1, FONT, right=True)

    txt(f"Detection threshold {conf:.2f}   |   q quit   p pause   e edit dots", 14, 850, 0.34, MUTED)
    return p


def confirm_dialog(viewer, frame, lines):
    img = (frame * 0.25).astype(np.uint8)
    k = ui_scale(frame)
    y = frame.shape[0] // 2 - int(30 * k)
    for i, (txt_, col) in enumerate(lines):
        put(img, txt_, (int(40 * k), y + int(46 * k) * i), k, col, 0.9, 2)
    viewer.show(img)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("y"), ord("Y")):
            return True
        if key in (ord("n"), ord("N"), 27, ord("q")):
            return False


def write_tracker_yaml(conf):
    path = os.path.abspath("botsort_custom.yaml")
    with open(path, "w") as f:
        f.write(f"""tracker_type: botsort
track_high_thresh: {conf}
track_low_thresh: 0.2
new_track_thresh: {conf}
track_buffer: 90
match_thresh: 0.8
fuse_score: True
gmc_method: none
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
model: auto
""")
    return path


def apply_conf(model, conf):
    try:
        for t in model.predictor.trackers:
            t.args.track_high_thresh = conf
            t.args.new_track_thresh = conf
        return True
    except Exception:
        return False


# ---------------------------------- MAIN ------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO_PATH)
    ap.add_argument("--out", default=OUTPUT_PATH)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--stride", type=int, default=1, help="har N-th frame process karo (2 = 2x tez)")
    ap.add_argument("--start", default=None, help='video ka asli start time "YYYY-MM-DD HH:MM:SS" (default: abhi)')
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--edit", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    stride = max(1, args.stride)

    args.edit = args.edit or EDIT_ZONES_ON_START
    if RESET_ON_START:
        args.reset = args.yes = True
    if args.reset:
        files = [ZONES_FILE] + sorted(glob.glob("stats_*.json")) + sorted(glob.glob("dwell_log*.csv"))
        files = [f for f in files if os.path.exists(f)]
        print("[i] Ye files delete hongi:", ", ".join(files) if files else "(koi nahi)")
        if files and not args.yes and input("Sara data clear karna hai? 'yes' likho: ").strip().lower() != "yes":
            raise SystemExit("Reset cancel.")
        for f in files:
            os.remove(f)
        print("[i] Sara data clear ho gaya.")
        args.setup = True

    reader = VideoReader(args.video)
    W, H, fps = reader.W, reader.H, reader.fps
    PW = panel_w(H)
    CW = W + PW
    print(f"[i] Video: {args.video} | {W}x{H} | {fps:.1f} fps | {reader.total} frames (~{fmt(reader.total / fps)})")

    out_path = args.out or os.path.splitext(args.video)[0] + "_output.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (CW, H))
    if not writer.isOpened():
        raise SystemExit(f"[!] Output video nahi ban pa rahi: {out_path} (folder/permission check karo)")
    print(f"[i] Output video: {out_path}")

    base_ts = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S").timestamp() if args.start else time.time()

    first = reader.first_frame()
    edited = False
    if args.setup and os.path.exists(ZONES_FILE):
        os.remove(ZONES_FILE)
    pts = None if args.setup else load_zones(W, H)
    if pts is None:
        print(f"[i] {NUM_ZONES} dots lagao (har zone ka ek dot)")
        pts = draw_zones(first, args.scale)
    else:
        print(f"[i] Loaded {NUM_ZONES} dots from {ZONES_FILE} (dobara lagane ke liye --setup)")
        if args.edit:
            new_pts = draw_zones(first, args.scale, existing=pts)
            if new_pts is not None and new_pts != pts:
                pts, edited = new_pts, True
    zones = [Zone(i, p) for i, p in enumerate(pts)]

    import torch
    use_half = torch.cuda.is_available()
    model = YOLO(args.model)
    stats = Stats(datetime.fromtimestamp(base_ts).strftime("%Y-%m-%d"))
    if edited:
        stats.reset(keep_zones=True)
    idm = IdManager(stats.data.get("next_id", 1))
    tracker = VisitTracker()
    conf = args.conf
    tracker_cfg = write_tracker_yaml(conf)
    conf_applied = conf

    win = "Vehicle Dwell Monitor"
    viewer = None if args.no_show else Viewer(win, CW, H, args.scale)
    print("[i] Running. q=quit p=pause e=move dots r=redraw x=clear ALL [ ]=threshold")

    def save_all():
        stats.data["next_id"] = idm.next_id
        stats.save()

    now = base_ts
    last_save = time.time()
    t_start = time.time()
    try:
        while True:
            frame, fidx = reader.read(stride)
            if frame is None:
                print("[i] Video khatam.")
                break
            vsec = fidx / fps
            now = base_ts + vsec
            clean = frame.copy()

            res = model.track(frame, persist=True, conf=min(DET_CONF, conf), iou=0.5, imgsz=args.imgsz,
                              classes=VEHICLE_CLASSES, tracker=tracker_cfg,
                              agnostic_nms=True, half=use_half, verbose=False)[0]
            if conf != conf_applied and apply_conf(model, conf):
                conf_applied = conf

            dets, in_zone = [], {}
            if res.boxes is not None and res.boxes.id is not None and len(res.boxes):
                ids = res.boxes.id.int().tolist()
                boxes = res.boxes.xyxy.tolist()
                clss = res.boxes.cls.int().tolist()
                cfs = res.boxes.conf.tolist()
                min_side = MIN_BOX_FRAC * W
                keep = [n for n in range(len(ids))
                        if boxes[n][2] - boxes[n][0] >= min_side and boxes[n][3] - boxes[n][1] >= min_side]
                taken = {idm.raw2vid[ids[n]] for n in keep if ids[n] in idm.raw2vid}
                for n in keep:
                    vid = idm.assign(ids[n], boxes[n], now, taken)
                    taken.add(vid)
                    dets.append((vid, res.names[clss[n]], boxes[n], cfs[n]))
                in_zone = assign_dots(dets, zones, {v: st["zone"] for v, st in tracker.s.items()})
            idm.prune(now)

            handle_events(tracker.step(in_zone, now, idm.hold_sec), stats)

            if time.time() - last_save > 30:
                save_all()
                last_save = time.time()

            # ---------- drawing ----------
            k = ui_scale(frame)
            draw_zone_circles(frame, zones, tracker, now, k)
            for vid, cname, box, cf in dets:
                x1, y1, x2, y2 = [int(v) for v in box]
                st = tracker.s.get(vid)
                if st and st["active"]:
                    col = RED
                    label = f"V{vid} {cname} | Z{st['zone'] + 1} {fmt(now - st['start'])}"
                    th = max(2, int(3 * k))
                else:
                    col = (200, 170, 90)
                    label = f"V{vid} {cname} {cf:.2f}"
                    th = max(1, int(2 * k))
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, th)
                tag(frame, label, x1, y1 - 4, k, col, (255, 255, 255), 0.5)

            seen = {d[0] for d in dets}
            for vid, st in tracker.s.items():
                if st["active"] and vid not in seen and vid in idm.last:
                    bx_, _ = idm.last[vid]
                    x1, y1, x2, y2 = [int(v) for v in bx_]
                    dashed_rect(frame, (x1, y1), (x2, y2), AMBER, max(2, int(2 * k)))
                    tag(frame, f"V{vid} LOST {int(now - st['last'])}s | Z{st['zone'] + 1} {fmt(now - st['start'])}",
                        x1, y1 - 4, k, AMBER, (0, 0, 0), 0.5)

            frac = (fidx / reader.total) if reader.total else 0.0
            panel = build_panel(H, stats, tracker, zones, now, vsec, frac, conf)
            canvas = np.hstack([frame, panel])
            writer.write(canvas)

            if fidx % (30 * stride) < stride:
                el = time.time() - t_start
                print(f"[i] {frac * 100:.0f}% | frame {fidx}/{reader.total} | {fidx / max(el, 1e-6):.1f} fps")

            if viewer:
                viewer.show(canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    while True:
                        k2 = cv2.waitKey(50) & 0xFF
                        if k2 in (ord("p"), ord(" ")):
                            break
                        if k2 == ord("q"):
                            raise KeyboardInterrupt
                        viewer.handle_key(k2)
                        viewer.show(canvas)
                elif viewer.handle_key(key):
                    pass
                elif key == ord("]"):
                    conf = min(0.95, round(conf + 0.05, 2))
                    print(f"[i] Threshold = {conf:.2f}")
                elif key == ord("["):
                    conf = max(0.10, round(conf - 0.05, 2))
                    print(f"[i] Threshold = {conf:.2f}")
                elif key == ord("e"):
                    prev_scale = viewer.scale
                    old = [list(z.pt) for z in zones]
                    viewer.close()
                    new_pts = draw_zones(clean, prev_scale, existing=old)
                    if new_pts is not None and new_pts != old:
                        zones = [Zone(i, p) for i, p in enumerate(new_pts)]
                        tracker.s.clear()
                        stats.reset(keep_zones=True)
                        idm = IdManager(1)
                    viewer = Viewer(win, CW, H, prev_scale)
                elif key == ord("x"):
                    if confirm_dialog(viewer, canvas, [
                            ("SARA DATA CLEAR karna hai?", (0, 255, 255)),
                            ("(zones, counts, logs sab delete)", (255, 255, 255)),
                            ("Y = haan, clear karo     N = nahi", (255, 255, 255))]):
                        prev_scale = viewer.scale
                        viewer.close()
                        tracker.s.clear()
                        stats.reset()
                        idm = IdManager(1)
                        pts = draw_zones(clean, prev_scale)
                        zones = [Zone(i, p) for i, p in enumerate(pts)]
                        viewer = Viewer(win, CW, H, prev_scale)
                elif key == ord("r"):
                    prev_scale = viewer.scale
                    viewer.close()
                    pts = draw_zones(clean, prev_scale)
                    zones = [Zone(i, p) for i, p in enumerate(pts)]
                    handle_events(tracker.close_all(), stats)
                    viewer = Viewer(win, CW, H, prev_scale)
    except KeyboardInterrupt:
        pass
    finally:
        handle_events(tracker.close_all(), stats)
        save_all()
        writer.release()
        reader.release()
        cv2.destroyAllWindows()
        print(f"[i] Stopped. Unique vehicles: {stats.unique} | zone visits: {stats.visits}")
        print(f"[i] Output video saved: {out_path}")
        print(f"[i] Log: {os.path.abspath(LOG_FILE)} | Stats: {os.path.abspath(stats._path())}")


if __name__ == "__main__":
    main()