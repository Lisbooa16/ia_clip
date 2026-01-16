import sys
import json
import gc
from faster_whisper import WhisperModel

def main():
    video_path = sys.argv[1]
    output_path = sys.argv[2]
    language = sys.argv[3] if len(sys.argv) > 3 else "auto"

    model = WhisperModel(
        "small",
        device="cuda",
        compute_type="float16"
    )

    kw = {}
    if language != "auto":
        kw["language"] = language

    segments_iter, _ = model.transcribe(
        video_path,
        word_timestamps=True,
        vad_filter=True,
        **kw
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write('{"segments":[\n')
        first = True
        count = 0

        for s in segments_iter:
            seg = {
                "start": float(s.start),
                "end": float(s.end),
                "text": (s.text or "").strip(),
                "words": [
                    {
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": (w.word or "").strip(),
                    }
                    for w in (s.words or [])
                ],
            }

            if not first:
                f.write(",\n")
            json.dump(seg, f, ensure_ascii=False)
            first = False
            count += 1

        f.write("\n]}")

    print(f"[WHISPER] OK ({count} segmentos)")
    del model
    gc.collect()
    sys.exit(0)

if __name__ == "__main__":
    main()
