from collections import defaultdict


def build_focus_timeline(
    faces_tracked,
    transcript,
    min_focus_duration=0.8
):
    from collections import defaultdict

    faces_by_id = defaultdict(list)
    for f in faces_tracked:
        faces_by_id[f["face_id"]].append(f)

    timeline = []
    last_face = None

    for seg in transcript.get("segments", []):
        start = round(float(seg["start"]), 2)
        end = round(float(seg["end"]), 2)
        duration = end - start

        best_face = None
        best_count = 0

        for face_id, items in faces_by_id.items():
            count = sum(
                1 for f in items
                if start <= f["time"] <= end
            )
            if count > best_count:
                best_count = count
                best_face = face_id

        chosen = best_face or last_face
        if chosen is None:
            continue

        if timeline and timeline[-1]["face_id"] == chosen:
            timeline[-1]["end"] = end
        else:
            if duration >= min_focus_duration:
                timeline.append({
                    "start": start,
                    "end": end,
                    "face_id": chosen
                })

        last_face = chosen

    return timeline
