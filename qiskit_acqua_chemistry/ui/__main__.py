# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import sys
import os
import logging
import tkinter as tk

qiskit_acqua_chemistry_directory = os.path.dirname(os.path.realpath(__file__))
qiskit_acqua_chemistry_directory = os.path.join(qiskit_acqua_chemistry_directory,'../..')
sys.path.insert(0,'qiskit_acqua_chemistry')
sys.path.insert(0,qiskit_acqua_chemistry_directory)
# hack untils qiskit-acqua is installable
qiskit_acqua_directory = os.path.join(qiskit_acqua_chemistry_directory,'../qiskit-acqua')
sys.path.append(qiskit_acqua_directory)
# ---

from qiskit_acqua_chemistry._logging import build_logging_config,set_logger_config
from _uipreferences import UIPreferences

if sys.platform == 'darwin':
    from Foundation import NSBundle
    bundle = NSBundle.mainBundle()
    if bundle:
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        info['CFBundleName'] = 'QISkit Acqua Chemistry'

root = tk.Tk()
root.withdraw()
root.update_idletasks()

preferences = UIPreferences()
geometry = preferences.get_geometry()
if geometry is None:
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    w = int(ws / 1.3)
    h = int(hs / 1.3)
    x = int(ws/2 - w/2)
    y = int(hs/2 - h/2)
    geometry = '{}x{}+{}+{}'.format(w,h,x,y)
    preferences.set_geometry(geometry)
    preferences.save()

root.geometry(geometry)

from qiskit_acqua_chemistry.preferences import Preferences

preferences = Preferences()
if preferences.get_logging_config() is None:
    logging_config = build_logging_config(['qischem','algorithms'],logging.INFO)
    preferences.set_logging_config(logging_config)
    preferences.save()

set_logger_config(preferences.get_logging_config())

from _mainview import MainView

view = MainView(root)
root.after(0, root.deiconify)
root.mainloop()

            

