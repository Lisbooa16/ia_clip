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

    return {
        "x": int(sum(b["x"] for b in boxes) / len(boxes)),
        "y": int(sum(b["y"] for b in boxes) / len(boxes)),
        "w": int(sum(b["w"] for b in boxes) / len(boxes)),
        "h": int(sum(b["h"] for b in boxes) / len(boxes)),
    }


def compute_vertical_crop(face_box, frame_w, frame_h):
    cx = face_box["x"] + face_box["w"] // 2

    crop_w = int(frame_h * 9 / 16)
    crop_h = frame_h

    x = max(0, min(cx - crop_w // 2, frame_w - crop_w))
    y = 0

    return {
        "x": int(x),
        "y": int(y),
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
