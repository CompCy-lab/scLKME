# has to be done at the end, after everything has been imported
import sys

from . import io
from . import tools as tl
from . import utils

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tl"]})
