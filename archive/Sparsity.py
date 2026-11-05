import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

@dataclass(frozen=True)
class Layout:
    n_cue: int = 100
    n_target: int = 100
    n_context: int = 50
