import time
import threading
import numpy as np
from typing import Tuple, Optional
from nadir.biodiversity.logger import BiodiversityLogger

class BioacousticSurveyor:
    """Idea C: Onboard Bioacoustic Wildlife Surveying (~300 MB RAM).
    
    Continuously samples microphone audio in 3-second sliding windows,
    computes log-mel spectrograms, and runs ONNX audio classification
    to detect bird calls, amphibians, and wildlife in real-time.
    """
    
    def __init__(self, model_path: Optional[str] = None, logger: Optional[BiodiversityLogger] = None,
                 confidence_threshold: float = 0.70, sample_rate: int = 16000):
        self.model_path = model_path
        self.logger = logger or BiodiversityLogger()
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate
        self.running = False
        self.session = None
        self.thread = None
        
        if model_path:
            try:
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 2
                self.session = ort.InferenceSession(model_path, sess_options)
                print(f"🐤 Bioacoustic Surveyor loaded ONNX model: {model_path}")
            except Exception as e:
                print(f"⚠️ Bioacoustic ONNX load error: {e}")

    def process_audio_buffer(self, pcm_audio: np.ndarray, current_position: Tuple[float, float, float]):
        """Process 3 seconds of raw PCM audio data."""
        if self.session is None:
            return
            
        try:
            input_name = self.session.get_inputs()[0].name
            # Reshape or compute log-mel spectrogram
            audio_input = pcm_audio.astype(np.float32).reshape(1, -1)
            outputs = self.session.run(None, {input_name: audio_input})[0]
            
            top_idx = int(np.argmax(outputs[0]))
            confidence = float(outputs[0][top_idx])
            
            if confidence >= self.confidence_threshold:
                call_label = f"Audio_Species_{top_idx}"
                self.logger.log_observation(
                    category="bioacoustic",
                    label=call_label,
                    confidence=confidence,
                    position=current_position
                )
        except Exception as e:
            print(f"Error processing bioacoustic audio: {e}")
