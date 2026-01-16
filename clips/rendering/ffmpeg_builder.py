from clips.media.crop_logic import compute_vertical_crop


def build_crop_filters(faces, frame_w, frame_h):
    filters = []

    for i, f in enumerate(faces):
        t_start = f["time"]
        t_end = faces[i + 1]["time"] if i + 1 < len(faces) else t_start + 1

        x1, y1, x2, y2 = compute_vertical_crop(f, frame_w, frame_h)

        filters.append(
            f"between(t,{t_start},{t_end})*crop={x2-x1}:{y2-y1}:{x1}:{y1}"
        )

    return "+".join(filters)
