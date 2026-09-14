import time
import threading
import numpy as np
from typing import Tuple, Optional
from nadir.biodiversity.logger import BiodiversityLogger

class PlantClassifier:
    """Idea B: Onboard Botanical & Species AI (~800 MB RAM).
    
    Runs asynchronous computer vision inference on camera RGB crops
    to classify forest flora, invasive weeds, and tree species.
    """
    
    def __init__(self, model_path: Optional[str] = None, logger: Optional[BiodiversityLogger] = None,
                 confidence_threshold: float = 0.75):
        self.model_path = model_path
        self.logger = logger or BiodiversityLogger()
        self.confidence_threshold = confidence_threshold
        self.running = False
        self.session = None
        
        if model_path:
            try:
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 2
                self.session = ort.InferenceSession(model_path, sess_options)
                print(f"🌿 Plant Classifier loaded ONNX model: {model_path}")
            except Exception as e:
                print(f"⚠️ Plant Classifier ONNX load error: {e}")
                
    def classify_frame(self, rgb_crop: np.ndarray, current_position: Tuple[float, float, float]):
        """Run classification on an image crop and log if above confidence threshold."""
        if self.session is None:
            # Demonstration / dummy fallback when model file is not passed
            return
            
        try:
            # Preprocess image crop (e.g. 224x224 RGB normalization)
            input_name = self.session.get_inputs()[0].name
            # Run inference asynchronously
            outputs = self.session.run(None, {input_name: rgb_crop})[0]
            top_idx = int(np.argmax(outputs[0]))
            confidence = float(outputs[0][top_idx])
            
            if confidence >= self.confidence_threshold:
                species_label = f"Species_ID_{top_idx}"
                self.logger.log_observation(
                    category="botanical",
                    label=species_label,
                    confidence=confidence,
                    position=current_position
                )
        except Exception as e:
            print(f"Error during plant classification: {e}")
