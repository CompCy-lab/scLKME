from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


@dataclass
class JobOutput:
    """
    Output signature of a multiple-processing job. This class provides
    an organized way to record and manipulate the output from a job.


    """

    jobid: int
    run_ok: bool
    result: Optional[Dict[str, Any]] = field(default_factory=dict)

    @property
    def result_keys(self) -> Iterable[str]:
        """keys for the result"""
        return self.result.keys()

    def _as_dict(self, attr_name: str):
        """transform a class attribute into a dict"""
        attr = getattr(self, attr_name)
        if isinstance(attr, dict):
            return attr
        else:
            return {attr_name: attr}
