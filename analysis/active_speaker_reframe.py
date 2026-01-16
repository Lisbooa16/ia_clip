import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class Track:
    track_id: int
    center_x: float
    center_y: float
    w: float
    h: float
    prev_mouth: Optional[np.ndarray]
    last_seen: int


def _resize_for_detection(frame: np.ndarray, target_width: int) -> Tuple[np.ndarray, float, float]:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame, 1.0, 1.0
    scale = target_width / w
    resized = cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, 1.0 / scale, 1.0 / scale


def _mouth_roi(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = bbox
    mx = int(x + 0.2 * w)
    my = int(y + 0.6 * h)
    mw = int(0.6 * w)
    mh = int(0.3 * h)
    mx = max(0, mx)
    my = max(0, my)
    mw = max(1, mw)
    mh = max(1, mh)
    roi = frame[my:my + mh, mx:mx + mw]
    if roi.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 32), interpolation=cv2.INTER_AREA)


def _match_tracks(
    detections: List[Tuple[float, float, float, float]],
    tracks: List[Track],
    frame_idx: int,
    next_id: int,
) -> Tuple[List[Track], int]:
    if not detections:
        return tracks, next_id

    det_centers = [
        (det[0] + det[2] / 2, det[1] + det[3] / 2, det[2], det[3])
        for det in detections
    ]
    used_tracks = set()
    updated_tracks = []

    for det_idx, (cx, cy, w, h) in enumerate(det_centers):
        best_track = None
        best_dist = None
        for track in tracks:
            if track.track_id in used_tracks:
                continue
            dist = abs(cx - track.center_x) + abs(cy - track.center_y)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_track = track
        if best_track is None or (best_dist is not None and best_dist > max(w, h)):
            track = Track(
                track_id=next_id,
                center_x=cx,
                center_y=cy,
                w=w,
                h=h,
                prev_mouth=None,
                last_seen=frame_idx,
            )
            next_id += 1
            updated_tracks.append(track)
        else:
            used_tracks.add(best_track.track_id)
            best_track.center_x = cx
            best_track.center_y = cy
            best_track.w = w
            best_track.h = h
            best_track.last_seen = frame_idx
            updated_tracks.append(best_track)

    return updated_tracks, next_id


def _pick_active_speaker(
    frame: np.ndarray,
    detections: List[Tuple[float, float, float, float]],
    tracks: List[Track],
    frame_idx: int,
) -> Tuple[Optional[Tuple[float, float, float, float]], List[Track]]:
    if not detections:
        return None, tracks

    track_map = {track.track_id: track for track in tracks}
    motion_scores = []
    for det in detections:
        cx = det[0] + det[2] / 2
        matched_track = None
        for track in tracks:
            if abs(track.center_x - cx) <= max(det[2], det[3]) * 0.5:
                matched_track = track
                break
        if not matched_track:
            motion_scores.append((0.0, det))
            continue
        mouth = _mouth_roi(frame, det)
        if matched_track.prev_mouth is not None:
            diff = cv2.absdiff(matched_track.prev_mouth, mouth)
            score = float(np.mean(diff))
        else:
            score = 0.0
        matched_track.prev_mouth = mouth
        matched_track.last_seen = frame_idx
        motion_scores.append((score, det))

    if motion_scores:
        motion_scores.sort(key=lambda item: (item[0], item[1][2] * item[1][3]), reverse=True)
        return motion_scores[0][1], tracks

    return detections[0], tracks


def _detect_faces(
    detector: mp.solutions.face_detection.FaceDetection,
    frame: np.ndarray,
    detect_width: int,
) -> List[Tuple[float, float, float, float]]:
    resized, sx, sy = _resize_for_detection(frame, detect_width)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    faces = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = bbox.xmin * resized.shape[1]
            y = bbox.ymin * resized.shape[0]
            w = bbox.width * resized.shape[1]
            h = bbox.height * resized.shape[0]
            faces.append((x * sx, y * sy, w * sx, h * sy))
    return faces


def reframe_video(
    input_path: str,
    output_path: str,
    detect_width: int = 640,
    detect_stride: int = 3,
    ema_alpha: float = 0.2,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w = int(frame_h * 9 / 16)
    crop_h = frame_h
    crop_w = min(crop_w, frame_w)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (crop_w, crop_h),
    )

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5,
    )

    buffer_frames = []
    last_detect_x = None
    last_smoothed_x = None
    tracks: List[Track] = []
    next_track_id = 0

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % detect_stride == 0:
            detections = _detect_faces(detector, frame, detect_width)
            tracks, next_track_id = _match_tracks(detections, tracks, frame_idx, next_track_id)
            active_det, tracks = _pick_active_speaker(frame, detections, tracks, frame_idx)
            target_x = None
            if active_det:
                target_x = active_det[0] + active_det[2] / 2
            elif last_detect_x is not None:
                target_x = last_detect_x
            else:
                target_x = frame_w / 2

            if last_detect_x is None:
                last_detect_x = target_x

            if buffer_frames:
                for i, buf_frame in enumerate(buffer_frames, start=1):
                    interp = last_detect_x + (target_x - last_detect_x) * (i / (len(buffer_frames) + 1))
                    last_smoothed_x = interp if last_smoothed_x is None else (
                        ema_alpha * interp + (1 - ema_alpha) * last_smoothed_x
                    )
                    _write_cropped(writer, buf_frame, last_smoothed_x, crop_w, crop_h, frame_w)
                buffer_frames.clear()

            last_smoothed_x = target_x if last_smoothed_x is None else (
                ema_alpha * target_x + (1 - ema_alpha) * last_smoothed_x
            )
            _write_cropped(writer, frame, last_smoothed_x, crop_w, crop_h, frame_w)
            last_detect_x = target_x
        else:
            buffer_frames.append(frame)

        frame_idx += 1

    if buffer_frames:
        fallback_x = last_detect_x if last_detect_x is not None else frame_w / 2
        for buf_frame in buffer_frames:
            last_smoothed_x = fallback_x if last_smoothed_x is None else (
                ema_alpha * fallback_x + (1 - ema_alpha) * last_smoothed_x
            )
            _write_cropped(writer, buf_frame, last_smoothed_x, crop_w, crop_h, frame_w)

    detector.close()
    cap.release()
    writer.release()


def _write_cropped(
    writer: cv2.VideoWriter,
    frame: np.ndarray,
    center_x: float,
    crop_w: int,
    crop_h: int,
    frame_w: int,
) -> None:
    cx = max(crop_w / 2, min(center_x, frame_w - crop_w / 2))
    x1 = int(round(cx - crop_w / 2))
    x2 = x1 + crop_w
    x1 = max(0, min(x1, frame_w - crop_w))
    x2 = x1 + crop_w
    cropped = frame[:crop_h, x1:x2]
    writer.write(cropped)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-reframe video around active speaker.")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--detect-width", type=int, default=640, help="Detection width")
    parser.add_argument("--detect-stride", type=int, default=3, help="Frame skip stride")
    parser.add_argument("--ema-alpha", type=float, default=0.2, help="EMA smoothing alpha")
    args = parser.parse_args()

    reframe_video(
        input_path=args.input,
        output_path=args.output,
        detect_width=args.detect_width,
        detect_stride=args.detect_stride,
        ema_alpha=args.ema_alpha,
    )


if __name__ == "__main__":
    main()
