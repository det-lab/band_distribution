"""band_distribution: Neganov-Luke phonon / ionization-charge band PDF and
event simulator for cryogenic detector nuclear-recoil (NR) and
electron-recoil (ER) bands.

PpqPDF and generate_events/generate_NR_events/generate_ER_events are the
two public entrypoints -- see their own docstrings (ppq_pdf.py,
event_generator.py). Everything under internal/ and cli/ is either what
those two are built from or infrastructure for maintaining this library,
and carries no stability promise.

PpqPDF has a hard, unconditional dependency on the compiled Fortran
library; event_generator.py has none. Importing this package never
touches the library -- PpqPDF only requires it be loadable (see
LD_LIBRARY_PATH/DYLD_LIBRARY_PATH in internal/_ppqfort_bindings.py) once
you construct one, and fails there, not on the first likelihood call deep
into a fit.
"""

from .event_generator import generate_ER_events, generate_events, generate_NR_events
from .ppq_pdf import PpqPDF

__all__ = ["PpqPDF", "generate_events", "generate_NR_events", "generate_ER_events"]

try:
    from importlib.metadata import PackageNotFoundError, version
    # The installed distribution name ("phonon-charge-pdf" on PyPI)
    # deliberately differs from this package's import name
    # (band_distribution) -- see pyproject.toml and the README's
    # "Installing the Python package" section -- so it has to be
    # hardcoded here rather than derived from __name__.
    __version__ = version("phonon-charge-pdf")
except PackageNotFoundError:
    __version__ = "0+unknown"
