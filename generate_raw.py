from pathlib import Path
import random
import struct
import zlib

import numpy as np
import pandas as pd


SEED = 76241
WIDTH = 720
HEIGHT = 480
N_SCENES = 56

COLORS = {
    "wrap": (190, 208, 214),
    "tray": (52, 61, 68),
    "well": (105, 116, 124),
    "well_dark": (35, 41, 48),
    "grid": (139, 156, 164),
    "white": (244, 246, 247),
    "black": (10, 12, 14),
    "gray": (155, 164, 170),
    "steel": (207, 216, 222),
    "forceps": (66, 214, 230),
    "clamp": (52, 206, 118),
    "scalpel": (237, 73, 72),
    "syringe": (170, 108, 246),
    "gauze": (246, 226, 110),
    "blue": (70, 146, 250),
    "brown": (156, 96, 44),
    "pink": (248, 122, 192),
    "green": (52, 206, 118),
    "amber": (255, 181, 54),
    "red": (238, 54, 54),
    "orange": (255, 132, 38),
    "yellow": (250, 228, 72),
}

INSTRUMENTS = ["forceps", "clamp", "scalpel", "syringe", "gauze"]
INDICATORS = ["blue", "brown", "pink", "green", "amber"]
ZONES = ["upper-left", "upper-right", "center", "lower-left", "lower-right"]
DIRECTIONS = ["left", "right", "up", "down", "diagonal"]

FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["10010", "10010", "10010", "11111", "00010", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01111", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "11110"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}

SEGMENTS = {
    0: "abcedf",
    1: "bc",
    2: "abged",
    3: "abgcd",
    4: "fgbc",
    5: "afgcd",
    6: "afgecd",
    7: "abc",
    8: "abcdefg",
    9: "abfgcd",
}


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: Path, img: np.ndarray) -> None:
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].astype(np.uint8).tobytes() for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b"")
    path.write_bytes(png)


def color(name: str) -> np.ndarray:
    return np.array(COLORS[name], dtype=np.float32)


def rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, c: str | tuple[int, int, int]) -> None:
    col = np.array(COLORS[c] if isinstance(c, str) else c, dtype=np.uint8)
    x0, x1 = max(0, x0), min(WIDTH, x1)
    y0, y1 = max(0, y0), min(HEIGHT, y1)
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = col


def blend_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, c: str, alpha: float) -> None:
    x0, x1 = max(0, x0), min(WIDTH, x1)
    y0, y1 = max(0, y0), min(HEIGHT, y1)
    if x0 >= x1 or y0 >= y1:
        return
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1].astype(np.float32) * (1 - alpha) + color(c) * alpha).clip(0, 255).astype(np.uint8)


def circle(img: np.ndarray, cx: int, cy: int, r: int, c: str) -> None:
    x0, x1 = max(0, cx - r), min(WIDTH, cx + r + 1)
    y0, y1 = max(0, cy - r), min(HEIGHT, cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= r * r
    patch = img[y0:y1, x0:x1]
    patch[mask] = np.array(COLORS[c], dtype=np.uint8)


def line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, c: str, thickness: int = 3) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for t in np.linspace(0, 1, steps + 1):
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        circle(img, x, y, thickness, c)


def text(img: np.ndarray, x: int, y: int, s: int, msg: str, c: str = "white") -> None:
    cursor = x
    for ch in msg.upper():
        glyph = FONT.get(ch, FONT[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    rect(img, cursor + gx * s, y + gy * s, cursor + (gx + 1) * s, y + (gy + 1) * s, c)
        cursor += 6 * s


def seg_boxes(x: int, y: int, s: int) -> dict[str, tuple[int, int, int, int]]:
    t = max(2, s // 2)
    w = 5 * s
    h = 9 * s
    return {
        "a": (x + t, y, x + w - t, y + t),
        "b": (x + w - t, y + t, x + w, y + h // 2 - t),
        "c": (x + w - t, y + h // 2 + t, x + w, y + h - t),
        "d": (x + t, y + h - t, x + w - t, y + h),
        "e": (x, y + h // 2 + t, x + t, y + h - t),
        "f": (x, y + t, x + t, y + h // 2 - t),
        "g": (x + t, y + h // 2 - t // 2, x + w - t, y + h // 2 + t // 2),
    }


def seven_digit(img: np.ndarray, x: int, y: int, digit: int, s: int, c: str) -> None:
    boxes = seg_boxes(x, y, s)
    for seg in SEGMENTS[int(digit)]:
        rect(img, *boxes[seg], c)


def seven_number(img: np.ndarray, x: int, y: int, value: int, s: int, c: str, digits: int = 2) -> None:
    label = str(value).zfill(digits)
    for i, ch in enumerate(label):
        seven_digit(img, x + i * (6 * s), y, int(ch), s, c)


def make_options(answer: str, candidates: list[str], rng: random.Random) -> tuple[dict[str, str], str]:
    values = [answer] + [c for c in candidates if c != answer]
    values = values[:5]
    rng.shuffle(values)
    letters = ["A", "B", "C", "D", "E"]
    opts = dict(zip(letters, values))
    return opts, next(k for k, v in opts.items() if v == answer)


def split_group(scene_idx: int, rng: random.Random) -> tuple[str, str, str, str]:
    if scene_idx < 34:
        return "standard_holdout", "standard", "normal", rng.choice(["low", "medium"])
    if scene_idx < 39:
        return "ood_low_light", "low_light", "low_light", "medium"
    if scene_idx < 44:
        return "ood_blue_cast", "blue_cast", "blue_cast", "medium"
    if scene_idx < 49:
        return "ood_glare", "glare", "glare", "medium"
    if scene_idx < 53:
        return "ood_wrap_occlusion", "wrap_occlusion", "normal", "high"
    return "ood_dense_clutter", "dense_clutter", "normal", "high"


def slot_boxes(layout: str) -> dict[int, tuple[int, int, int, int]]:
    if layout == "wide_grid":
        x0, y0, w, h, gapx, gapy = 45, 130, 150, 92, 30, 32
    elif layout == "compact_grid":
        x0, y0, w, h, gapx, gapy = 92, 136, 132, 88, 22, 28
    elif layout == "offset_grid":
        x0, y0, w, h, gapx, gapy = 64, 122, 138, 90, 24, 42
    else:
        x0, y0, w, h, gapx, gapy = 64, 130, 140, 90, 26, 34
    boxes = {}
    for idx in range(6):
        row, col = divmod(idx, 3)
        xo = x0 + col * (w + gapx) + (14 if layout == "offset_grid" and row == 1 else 0)
        yo = y0 + row * (h + gapy)
        boxes[idx + 1] = (xo, yo, xo + w, yo + h)
    return boxes


def scene_trace(scene_idx: int, rng: random.Random) -> dict:
    group, axis, lighting, clutter = split_group(scene_idx, rng)
    layout = rng.choice(["standard_grid", "wide_grid", "compact_grid", "offset_grid"])
    slots = {i: None for i in range(1, 7)}
    empty_slot = rng.randint(1, 6)
    unique_item = rng.choice(INSTRUMENTS)
    pool = [unique_item] + rng.choices([item for item in INSTRUMENTS if item != unique_item], k=4)
    insert_slots = [s for s in range(1, 7) if s != empty_slot]
    rng.shuffle(insert_slots)
    for slot, item in zip(insert_slots, pool):
        slots[slot] = item
    item_counts = {item: sum(1 for v in slots.values() if v == item) for item in INSTRUMENTS}
    single_items = [item for item, count in item_counts.items() if count == 1]
    query_item = rng.choice(single_items)
    count_item = rng.choice(INSTRUMENTS)
    route = rng.sample(["1", "2", "3", "4", "5"], 5)
    route_query = rng.choice(route[:-1])
    return {
        "scene_id": f"tray_{scene_idx:03d}",
        "image": f"images/tray_{scene_idx:03d}.png",
        "split_group": group,
        "ood_axis": axis,
        "layout_family": layout,
        "lighting_condition": lighting,
        "clutter_level": clutter,
        "slots": slots,
        "empty_slot": empty_slot,
        "query_item": query_item,
        "count_item": count_item,
        "indicator": rng.choice(INDICATORS),
        "tag_code": rng.randint(10, 98),
        "breach_zone": rng.choice(ZONES),
        "arrow_direction": rng.choice(DIRECTIONS),
        "route": route,
        "route_query": route_query,
    }


def draw_instrument(img: np.ndarray, item: str, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    if item == "forceps":
        line(img, cx - 38, cy + 24, cx + 30, cy - 20, "forceps", 5)
        line(img, cx - 22, cy + 24, cx + 40, cy - 18, "forceps", 4)
        circle(img, cx + 36, cy - 19, 4, "steel")
    elif item == "clamp":
        line(img, cx - 42, cy, cx + 42, cy, "clamp", 5)
        circle(img, cx - 43, cy + 18, 12, "clamp")
        circle(img, cx - 43, cy + 18, 6, "well_dark")
        circle(img, cx + 43, cy + 18, 12, "clamp")
        circle(img, cx + 43, cy + 18, 6, "well_dark")
    elif item == "scalpel":
        rect(img, cx - 44, cy - 7, cx + 22, cy + 8, "scalpel")
        line(img, cx + 20, cy, cx + 50, cy - 12, "steel", 7)
        line(img, cx + 20, cy, cx + 50, cy + 10, "steel", 4)
    elif item == "syringe":
        rect(img, cx - 42, cy - 10, cx + 30, cy + 11, "syringe")
        rect(img, cx + 30, cy - 4, cx + 52, cy + 5, "steel")
        line(img, cx - 54, cy, cx - 42, cy, "steel", 4)
        rect(img, cx - 58, cy - 15, cx - 52, cy + 16, "syringe")
    elif item == "gauze":
        rect(img, cx - 34, cy - 25, cx + 34, cy + 25, "gauze")
        for xx in range(cx - 28, cx + 35, 14):
            line(img, xx, cy - 25, xx, cy + 25, "white", 1)
        for yy in range(cy - 20, cy + 26, 10):
            line(img, cx - 34, yy, cx + 34, yy, "white", 1)


def draw_arrow(img: np.ndarray, box: tuple[int, int, int, int], direction: str) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    vectors = {
        "left": (-45, 0),
        "right": (45, 0),
        "up": (0, -34),
        "down": (0, 34),
        "diagonal": (36, -28),
    }
    dx, dy = vectors[direction]
    line(img, cx, cy, cx + dx, cy + dy, "orange", 5)
    circle(img, cx + dx, cy + dy, 8, "orange")


def draw_scene(trace: dict, out_path: Path, np_rng: np.random.Generator) -> None:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:] = np.array(COLORS["wrap"], dtype=np.uint8)
    rect(img, 24, 24, 696, 456, "tray")
    for x in range(40, 690, 38):
        line(img, x, 36, x, 446, "grid", 1)
    for y in range(42, 450, 38):
        line(img, 36, y, 684, y, "grid", 1)

    boxes = slot_boxes(trace["layout_family"])
    for slot, box in boxes.items():
        x0, y0, x1, y1 = box
        rect(img, x0, y0, x1, y1, "well")
        rect(img, x0 + 6, y0 + 8, x1 - 6, y1 - 8, "well_dark")
        text(img, x0 + 8, y0 + 10, 2, str(slot), "white")
        item = trace["slots"][slot]
        if item is not None:
            draw_instrument(img, item, (x0 + 12, y0 + 10, x1 - 12, y1 - 8))

    arrow_slot = next((slot for slot, item in trace["slots"].items() if item is not None), 1)
    x0, y0, x1, y1 = boxes[arrow_slot]
    blend_rect(img, x0 - 4, y0 - 4, x1 + 4, y1 + 4, "orange", 0.24)
    draw_arrow(img, (x0 + 12, y0 + 12, x1 - 12, y1 - 12), trace["arrow_direction"])

    rect(img, 520, 64, 656, 112, "well_dark")
    text(img, 530, 72, 2, "STERILE", "white")
    rect(img, 530, 94, 644, 106, trace["indicator"])

    rect(img, 282, 54, 430, 126, "black")
    text(img, 294, 60, 2, "TAG", "gray")
    seven_number(img, 330, 82, trace["tag_code"], 5, "green")

    rect(img, 258, 396, 468, 448, "black")
    text(img, 270, 402, 2, "CHECK", "gray")
    for i, step in enumerate(trace["route"]):
        seven_number(img, 288 + i * 32, 422, int(step), 3, "yellow", digits=1)
        if i < 4:
            line(img, 305 + i * 32, 435, 316 + i * 32, 435, "gray", 1)

    zone_points = {
        "upper-left": (105, 78),
        "upper-right": (616, 78),
        "center": (360, 238),
        "lower-left": (108, 420),
        "lower-right": (618, 420),
    }
    bx, by = zone_points[trace["breach_zone"]]
    for k in range(4):
        line(img, bx - 18 + k * 6, by - 8 + k * 3, bx - 8 + k * 6, by + 10 - k * 2, "red", 4)

    n_clutter = {"low": 18, "medium": 44, "high": 96}[trace["clutter_level"]]
    for _ in range(n_clutter):
        cx = int(np_rng.integers(28, WIDTH - 28))
        cy = int(np_rng.integers(28, HEIGHT - 28))
        c = str(np_rng.choice(["white", "gray", "steel", "blue", "amber"]))
        circle(img, cx, cy, int(np_rng.integers(1, 3)), c)

    if trace["lighting_condition"] == "low_light":
        img = (img.astype(np.float32) * 0.48).clip(0, 255).astype(np.uint8)
    elif trace["lighting_condition"] == "blue_cast":
        img[:, :, 2] = np.clip(img[:, :, 2].astype(np.int16) + 44, 0, 255)
        img[:, :, 0] = (img[:, :, 0].astype(np.float32) * 0.78).astype(np.uint8)
    elif trace["lighting_condition"] == "glare":
        for _ in range(4):
            gx = int(np_rng.integers(80, 600))
            gy = int(np_rng.integers(70, 350))
            blend_rect(img, gx, gy, gx + 120, gy + 30, "white", 0.48)

    if trace["ood_axis"] == "wrap_occlusion":
        blend_rect(img, 70, 200, 656, 236, "wrap", 0.82)
        blend_rect(img, 180, 334, 620, 362, "wrap", 0.70)

    noise = np_rng.normal(0, 5.0, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    write_png(out_path, img)


def add_row(rows: list[dict], trace: dict, qnum: int, query_type: str, question: str, answer_text: str, candidates: list[str], task_family: str, rng: random.Random) -> None:
    opts, answer_letter = make_options(answer_text, candidates, rng)
    rows.append(
        {
            "id": f"{trace['scene_id']}_q{qnum:02d}",
            "scene_id": trace["scene_id"],
            "image": trace["image"],
            "query_type": query_type,
            "question": question,
            "option_A": opts["A"],
            "option_B": opts["B"],
            "option_C": opts["C"],
            "option_D": opts["D"],
            "option_E": opts["E"],
            "answer": answer_letter,
            "answer_text": answer_text,
            "split_group": trace["split_group"],
            "ood_axis": trace["ood_axis"],
            "layout_family": trace["layout_family"],
            "lighting_condition": trace["lighting_condition"],
            "clutter_level": trace["clutter_level"],
            "task_family": task_family,
        }
    )


def question_rows(trace: dict, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    q = 0
    instrument_answer = next(f"Slot {slot}" for slot, item in trace["slots"].items() if item == trace["query_item"])
    add_row(rows, trace, q, "instrument_slot", f"Which compartment contains the {trace['query_item']}?", instrument_answer, [f"Slot {i}" for i in range(1, 7)], "inventory_check", rng)
    q += 1
    add_row(rows, trace, q, "empty_slot", "Which numbered compartment is empty?", f"Slot {trace['empty_slot']}", [f"Slot {i}" for i in range(1, 7)], "inventory_check", rng)
    q += 1
    add_row(rows, trace, q, "indicator_color", "What color is the sterility indicator strip?", trace["indicator"], INDICATORS, "sterility_check", rng)
    q += 1
    tag_candidates = [str(v) for v in rng.sample([n for n in range(10, 99) if n != trace["tag_code"]], 4)] + [str(trace["tag_code"])]
    add_row(rows, trace, q, "tag_code", "What two-digit tray tag is displayed?", str(trace["tag_code"]), tag_candidates, "text_reading", rng)
    q += 1
    add_row(rows, trace, q, "seal_breach", "Where is the red wrap breach mark located?", trace["breach_zone"], ZONES, "sterility_check", rng)
    q += 1
    add_row(rows, trace, q, "orientation_arrow", "Which direction does the orange orientation arrow point?", trace["arrow_direction"], DIRECTIONS, "spatial_reasoning", rng)
    q += 1
    count = sum(1 for item in trace["slots"].values() if item == trace["count_item"])
    add_row(rows, trace, q, "instrument_count", f"How many {trace['count_item']} items are visible?", str(count), ["0", "1", "2", "3", "4"], "inventory_check", rng)
    q += 1
    route_answer = trace["route"][trace["route"].index(trace["route_query"]) + 1]
    add_row(rows, trace, q, "checklist_order", f"In the checklist strip, which step comes immediately after step {trace['route_query']}?", route_answer, ["1", "2", "3", "4", "5"], "sequence_memory", rng)
    return rows


def main() -> None:
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)
    root = Path(__file__).resolve().parent
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for scene_idx in range(N_SCENES):
        trace = scene_trace(scene_idx, rng)
        draw_scene(trace, root / trace["image"], np_rng)
        rows.extend(question_rows(trace, rng))

    df = pd.DataFrame(rows)
    df.to_csv(root / "questions.csv", index=False)
    print(f"Wrote {len(df)} rows and {N_SCENES} images to {root}")


if __name__ == "__main__":
    main()
