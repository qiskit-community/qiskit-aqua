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
import logging
import tkinter as tk
from ._aquaguiprovider import AquaGUIProvider
from ._mainview import MainView


def set_preferences_logging():
    """
    Update logging setting with latest external packages
    """
    from qiskit.aqua._logging import get_logging_level, build_logging_config, set_logging_config
    from qiskit_aqua_cmd.preferences import Preferences
    preferences = Preferences()
    logging_level = logging.INFO
    if preferences.get_logging_config() is not None:
        set_logging_config(preferences.get_logging_config())
        logging_level = get_logging_level()

    preferences.set_logging_config(build_logging_config(logging_level))
    preferences.save()

    set_logging_config(preferences.get_logging_config())


def main():
    if sys.platform == 'darwin':
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        if bundle:
            info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            info['CFBundleName'] = 'Qiskit Aqua'

    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()

    guiProvider = AquaGUIProvider()
    preferences = guiProvider.create_uipreferences()
    geometry = preferences.get_geometry()
    if geometry is None:
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        w = int(ws / 1.3)
        h = int(hs / 1.3)
        x = int(ws / 2 - w / 2)
        y = int(hs / 2 - h / 2)
        geometry = '{}x{}+{}+{}'.format(w, h, x, y)
        preferences.set_geometry(geometry)
        preferences.save()

    root.geometry(geometry)

    MainView(root, guiProvider)
    root.after(0, root.deiconify)
    root.after(0, set_preferences_logging)
    root.mainloop()
