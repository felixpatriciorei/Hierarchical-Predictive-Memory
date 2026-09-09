"""Hierarchical Predictive Memory research package.

The repository was re-founded from the former HPM-Lite research testbed in
September 2026. Historical experiment artifacts retain the model identifier
``hpm_lite_v2`` for replay compatibility, while the maintained Python package
is now ``hpm``.
"""

from .hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model

# Clean public aliases for new code. Historical class names remain available
# because old evidence/configs refer to them.
HPMConfig = HpmLiteV2Config
HPMModel = HpmLiteV2Model

__all__ = ["HPMConfig", "HPMModel", "HpmLiteV2Config", "HpmLiteV2Model"]
__version__ = "0.1.0"
