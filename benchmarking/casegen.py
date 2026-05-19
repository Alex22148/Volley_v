from itertools import product


def build_case_grid(grid: dict) -> list[dict]:
    models = list(grid.get("models") or [])
    image_sizes = list(grid.get("image_sizes") or [])
    chunk_batch_sizes = list(grid.get("chunk_batch_sizes") or [])
    seq_lens = list(grid.get("seq_lens") or [])
    object_scales = list(grid.get("object_scales") or [])

    out = []
    for model_path, imgsz, chunk_bs, seq_len, scale in product(
        models, image_sizes, chunk_batch_sizes, seq_lens, object_scales
    ):
        out.append(
            {
                "model_path": str(model_path),
                "imgsz": int(imgsz),
                "chunk_batch_size": int(chunk_bs),
                "seq_len": int(seq_len),
                "object_scale": float(scale),
            }
        )
    return out

