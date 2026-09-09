# Migration from HPM-Lite-Memory-Model

## Repository name

Recommended new GitHub repository:

`felixpatriciorei/Hierarchical-Predictive-Memory`

The old repository should be archived/read-only rather than force-pushed into the new history.

## Python imports

New code:

```python
from hpm import HPMConfig, HPMModel
```

Historical class names remain available:

```python
from hpm import HpmLiteV2Config, HpmLiteV2Model
```

The model identifier `hpm_lite_v2` is temporarily retained in experiment configs for evidence
compatibility.

## Data/results

Do not copy the old `runs/`, `results/`, checkpoints, or `.git` directory into the new repo.
Keep large historical artifacts in a separate archive if desired.
