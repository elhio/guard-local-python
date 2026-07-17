# TODO: Core ML/heuristics (loads models, runs inference)
import asyncio
import numpy as np
from PIL import Image
import onnxruntime as ort


class LocalDetectorEngine:
    def __init__(self, model_path: str):
        # Load the ONNX model into memory once during initialization
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, image_path: str) -> np.ndarray:
        """Synchronous preprocessing (Pillow + Numpy replaces torchvision)"""
        # 1. Load image
        img = Image.open(image_path).convert("RGB")

        # 2. Resize and crop (example: 224x224)
        img = img.resize((224, 224))

        # 3. Convert to numpy array and scale to 0-1
        img_data = np.array(img).astype('float32') / 255.0

        # 4. Normalize (using standard ImageNet means/stds usually used in timm)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_data = (img_data - mean) / std

        # 5. ONNX expects channels first (B, C, H, W)
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)

        return img_data

    def _run_inference_sync(self, image_path: str) -> dict:
        """The blocking CPU work"""
        input_tensor = self._preprocess(image_path)

        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Parse output logic here
        confidence_score = float(outputs[0][0][1])  # Example parsing
        return {"status": "safe", "score": confidence_score}

    async def analyze_file_async(self, image_path: str) -> dict:
        """
        The async wrapper.
        This offloads the heavy CPU work to a separate thread so it
        does not block the main async event loop.
        """
        return await asyncio.to_thread(self._run_inference_sync, image_path)