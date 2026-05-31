from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


class CatDetector:
    def __init__(
        self,
        onnx_path: str | Path,
        imgsz: int = 640,
        conf: float = 0.25,
        class_names: tuple[str, ...] = ("cat",),
    ) -> None:
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_path: str | Path) -> list[dict[str, float | str]]:
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        x, scale, pad_x, pad_y = self._letterbox(img)
        x = (np.asarray(x, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        output = self.session.run(None, {self.input_name: x})[0]
        detections = self._normalize_output(output)

        results: list[dict[str, float | str]] = []
        for x1, y1, x2, y2, score, cls in detections:
            if float(score) < self.conf:
                continue

            xmin = (float(x1) - pad_x) / scale
            ymin = (float(y1) - pad_y) / scale
            xmax = (float(x2) - pad_x) / scale
            ymax = (float(y2) - pad_y) / scale

            xmin = max(0.0, min(float(orig_w), xmin))
            ymin = max(0.0, min(float(orig_h), ymin))
            xmax = max(0.0, min(float(orig_w), xmax))
            ymax = max(0.0, min(float(orig_h), ymax))

            if xmax <= xmin or ymax <= ymin:
                continue

            cls_idx = int(cls)
            cls_name = self.class_names[cls_idx] if 0 <= cls_idx < len(self.class_names) else str(cls_idx)
            results.append(
                {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "confidence": float(score),
                    "class": cls_name,
                }
            )

        return results

    def _letterbox(self, img: Image.Image) -> tuple[Image.Image, float, float, float]:
        orig_w, orig_h = img.size
        scale = min(self.imgsz / orig_w, self.imgsz / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        pad_x = (self.imgsz - new_w) / 2
        pad_y = (self.imgsz - new_h) / 2

        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))
        canvas.paste(resized, (int(round(pad_x - 0.1)), int(round(pad_y - 0.1))))
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def _normalize_output(output: np.ndarray) -> np.ndarray:
        output = np.asarray(output)
        if output.ndim == 3:
            output = output[0]
        if output.shape[-1] != 6:
            raise ValueError(f"Expected YOLO26 end-to-end output with 6 columns, got {output.shape}")
        return output.astype(np.float32, copy=False)
