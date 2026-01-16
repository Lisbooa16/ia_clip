def find_focus_face(focus_timeline, start, end):
    for block in focus_timeline:
        if block["start"] <= start <= block["end"]:
            return block["face_id"]
    return None


def average_face_box(faces_tracked, face_id, start, end):
    boxes = [
        f for f in faces_tracked
        if f["face_id"] == face_id and start <= f["time"] <= end
    ]

    if not boxes:
        return None

    print(
        "[FOCUS] 📦 "
        f"face_id={face_id} samples={len(boxes)} "
        f"window={start:.3f}-{end:.3f}s"
    )

    return {
        "x": int(sum(b["x"] for b in boxes) / len(boxes)),
        "y": int(sum(b["y"] for b in boxes) / len(boxes)),
        "w": int(sum(b["w"] for b in boxes) / len(boxes)),
        "h": int(sum(b["h"] for b in boxes) / len(boxes)),
    }


def stable_face_box(faces_tracked, face_id, start, end):
    boxes = [
        f for f in faces_tracked
        if f["face_id"] == face_id and start <= f["time"] <= end
    ]

    if not boxes:
        return None

    centers_x = sorted(b["x"] + b["w"] / 2 for b in boxes)
    centers_y = sorted(b["y"] + b["h"] / 2 for b in boxes)
    widths = [b["w"] for b in boxes]
    heights = [b["h"] for b in boxes]
    mid = len(centers_x) // 2
    center_x = centers_x[mid]
    center_y = centers_y[mid]
    max_w = max(widths)
    max_h = max(heights)
    x = center_x - max_w / 2
    y = center_y - max_h / 2

    print(
        "[FOCUS] 📦 "
        f"face_id={face_id} samples={len(boxes)} "
        f"center=({center_x:.1f},{center_y:.1f}) "
        f"size=({max_w:.1f},{max_h:.1f}) "
        f"window={start:.3f}-{end:.3f}s"
    )

    return {
        "x": int(round(x)),
        "y": int(round(y)),
        "w": int(round(max_w)),
        "h": int(round(max_h)),
    }


def compute_vertical_crop(face_box, frame_w, frame_h):
    x = float(face_box["x"])
    y = float(face_box["y"])
    w = float(face_box["w"])
    h = float(face_box["h"])

    left_pad = 0.18 * w
    right_pad = 0.18 * w
    top_pad = 0.5 * h
    bottom_pad = 0.2 * h

    exp_left = max(0.0, x - left_pad)
    exp_top = max(0.0, y - top_pad)
    exp_right = min(float(frame_w), x + w + right_pad)
    exp_bottom = min(float(frame_h), y + h + bottom_pad)

    face_center_x = x + w * 0.5
    face_center_y = y + h * 0.45

    crop_w = int(frame_h * 9 / 16)
    crop_h = frame_h

    desired_x = face_center_x - crop_w / 2
    desired_y = face_center_y - crop_h / 2
    min_x = exp_right - crop_w
    max_x = exp_left
    if min_x > max_x:
        min_x, max_x = max_x, min_x
    clamped = False
    if desired_x < min_x:
        desired_x = min_x
        clamped = True
    if desired_x > max_x:
        desired_x = max_x
        clamped = True

    desired_x = max(0.0, min(desired_x, frame_w - crop_w))
    desired_y = max(0.0, min(desired_y, frame_h - crop_h))
    if desired_x in (0.0, frame_w - crop_w):
        clamped = True
    if desired_y in (0.0, frame_h - crop_h):
        clamped = True

    print(
        "[CROP] 🧭 "
        f"raw=({x:.1f},{y:.1f},{w:.1f},{h:.1f}) "
        f"exp=({exp_left:.1f},{exp_top:.1f},{exp_right-exp_left:.1f},{exp_bottom-exp_top:.1f}) "
        f"crop=({desired_x:.1f},{desired_y:.1f},{crop_w},{crop_h}) "
        f"clamp={clamped} center_y={face_center_y:.1f}"
    )

    return {
        "x": int(round(desired_x)),
        "y": int(round(desired_y)),
        "w": crop_w,
        "h": crop_h,
    }


def focus_blocks_for_clip(focus_timeline, start, end):
    blocks = []
    min_block = 0.5
    last_face_id = None

    for b in focus_timeline:
        s = max(start, b["start"])
        e = min(end, b["end"])

        if s < e:
            face_id = b.get("face_id")
            if face_id is None and last_face_id is not None:
                face_id = last_face_id
            if face_id is not None:
                last_face_id = face_id
            blocks.append({
                "start": round(s, 3),
                "end": round(e, 3),
                "face_id": face_id,
            })

    if not blocks:
        return blocks

    merged = []
    for block in blocks:
        duration = block["end"] - block["start"]
        if merged:
            prev = merged[-1]
            if block["face_id"] == prev["face_id"]:
                prev["end"] = max(prev["end"], block["end"])
                continue
            if duration < min_block:
                prev["end"] = max(prev["end"], block["end"])
                continue
        if duration < min_block and merged:
            merged[-1]["end"] = max(merged[-1]["end"], block["end"])
            continue
        merged.append(block)

    for block in merged:
        print(
            "[FOCUS] 🧠 "
            f"{block['start']:.3f}-{block['end']:.3f}s "
            f"face_id={block['face_id']}"
        )

    return merged
