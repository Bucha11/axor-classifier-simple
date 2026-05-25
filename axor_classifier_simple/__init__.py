from axor_classifier_simple._version import get_version
from axor_classifier_simple.task_signal import TaskSignalClassifier, ModelNotTrainedError
from axor_classifier_simple.anomaly_detector import MLAnomalyDetector

__version__ = get_version("axor-classifier-simple")

__all__ = ["TaskSignalClassifier", "MLAnomalyDetector", "ModelNotTrainedError", "__version__"]
