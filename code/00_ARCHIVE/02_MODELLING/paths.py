# adding submodule files to the python path
# => https://stackoverflow.com/questions/38237284/setting-a-default-sys-path-for-ipython-notebook

import sys
import pathlib

current_path = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(current_path / '00_ARCHIVE' / '02_MODELLING' / 'models'))
sys.path.insert(0, str(current_path / '01_ANALYSIS'))
sys.path.insert(0, str(current_path / '02_MODELLING'))
sys.path.insert(0, str(current_path / '02_MODELLING' / 'models'))
