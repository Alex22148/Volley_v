# Vision Runtime Scaffold

Scaffold for:
- offline folder runner
- detector adapter: ultralytics / tensorrt
- debayer adapter: cpu_opencv / future_cuda_npp
- minimal preprocess
- tracker runtime
- runtime stats
- later: live backend integration

Next steps:
1. Fill ipc/messages.py
2. Fill adapters/detector_adapter.py
3. Fill adapters/debayer_adapter.py
4. Fill core/preprocess.py
5. Fill core/pipeline_runner.py
6. Fill apps/run_folder_inference.py
