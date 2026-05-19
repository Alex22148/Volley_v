# engines/

Drop the FP16, batch=4, **static** TensorRT engines here. The sweep
expects these exact filenames:

| filename                                    | inference H x W |
|---------------------------------------------|-----------------|
| `best__fp16_640_b4_static.engine`           | 640 x 640       |
| `best__fp16_960_b4_static.engine`           | 960 x 960       |
| `best__fp16_1280_b4_static.engine`          | 1280 x 1280     |
| `best__fp16_1088x1920_b4_static.engine`     | 1088 x 1920     |

If a file is missing, the sweep marks that row as `skipped` and keeps
going — partial reports are fine.

## Where do these come from?

They are exported in the main VolleyHub repo by
`tools/export_all_engine.py` / `new_res/tools/export_tensorrt_variants.py`.
This lab package does NOT re-export them — that requires the full
training/export environment.

If `tools/make_fastpath_lab_package.py --include-engines` was used to
build this package, the four engine files are already inside this
folder.

## Why FP16 batch=4 static?

* **Static**: the lab runner deliberately does not support dynamic-shape
  engines. This keeps the runner under 250 lines.
* **batch=4**: matches the four-camera packet shape of the production
  system, so the FPS numbers translate 1:1.
* **FP16**: the production system runs FP16 and so does this benchmark.

A 1088x1920 (height x width) Full HD-like engine is here precisely to
answer the question "could we just run inference at the camera
resolution?" — which is the most interesting hypothetical for the
tester.
