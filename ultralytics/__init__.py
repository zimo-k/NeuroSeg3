__version__ = '8.0.202'

# from neuroseg3-seg.models import RTDETR, SAM, YOLO
from .models import YOLO

from ultralytics.models import YOLO
# from neuroseg3-seg.models.fastsam import FastSAM
# from neuroseg3-seg.models.nas import NAS
from ultralytics.utils import SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download


# __all__ = '__version__', 'YOLO', 'NAS', 'SAM', 'FastSAM', 'RTDETR', 'checks', 'download', 'settings'
__all__ = '__version__', 'YOLO', 'checks', 'download', 'settings'
