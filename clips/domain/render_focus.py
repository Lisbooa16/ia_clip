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

    for b in focus_timeline:
        s = max(start, b["start"])
        e = min(end, b["end"])

        if s < e:
            blocks.append({
                "start": round(s, 3),
                "end": round(e, 3),
                "face_id": b["face_id"],
            })

    return blocks