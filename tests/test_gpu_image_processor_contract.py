import importlib

import numpy as np


def test_gpu_image_processor_imports_without_requiring_cuda():
    module = importlib.import_module("src.runtime_gpu.gpu_image_processor")
    assert hasattr(module, "GpuImageProcessor")


def test_image_processor_config_creation():
    from src.runtime_gpu.gpu_image_processor import ImageProcessorConfig

    cfg = ImageProcessorConfig(target_size=(32, 48), input_color="RGB", device="auto")

    assert cfg.target_hw() == (32, 48)
    cfg.validate()


def test_rgb_uint8_frame_to_normalized_bchw_tensor_cpu_safe():
    from src.runtime_gpu.gpu_image_processor import (
        GpuImageProcessor,
        ImageProcessorConfig,
    )

    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    frame[..., 0] = 255
    frame[..., 1] = 127
    frame[..., 2] = 64

    processor = GpuImageProcessor(
        ImageProcessorConfig(
            target_size=(8, 10),
            input_color="RGB",
            normalize=True,
            device="cpu",
        )
    )
    result = processor.process_frame(frame)
    tensor = result.tensor

    assert tuple(tensor.shape) == (1, 3, 8, 10)
    assert str(tensor.device) == "cpu"
    assert str(tensor.dtype).endswith("float32")
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    assert isinstance(result.stage_ms, dict)
    assert "total_ms" in result.stage_ms
    assert "resize_ms" in result.stage_ms
    assert result.layout == "BCHW"
    assert result.color_format == "RGB"
    assert result.source_shapes == [(12, 16, 3)]


def test_auto_cuda_falls_back_without_cuda_error():
    from src.runtime_gpu.gpu_image_processor import (
        GpuImageProcessor,
        ImageProcessorConfig,
    )

    frame = np.full((4, 4, 3), 128, dtype=np.uint8)
    processor = GpuImageProcessor(
        ImageProcessorConfig(
            target_size=4,
            input_color="RGB",
            normalize=True,
            device="auto",
            prefer_cuda=True,
        )
    )

    result = processor.process_frame(frame)

    assert tuple(result.tensor.shape) == (1, 3, 4, 4)
    assert result.device in ("cpu", "cuda", "cuda:0")
