# -*- coding: utf-8 -*-

# Copyright 2019 IBM.
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

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as tkfd
import os
from qiskit_aqua_ui import GUIProvider
from ._uipreferences import UIPreferences
from ._chemsectionpropertiesview import ChemSectionPropertiesView


class ChemistryGUIProvider(GUIProvider):
    """
    Chemistry GUIProvider
    """

    def __init__(self, controller):
        super().__init__(controller)
        self._preferences = None

    @property
    def title(self):
        """Return provider title."""
        return 'Qiskit Chemistry'

    @property
    def version(self):
        """Return provider version."""
        from qiskit_chemistry import __version__
        return __version__

    @property
    def help_hyperlink(self):
        """Return provider help hyperlink."""
        return 'http://qiskit.org/documentation/aqua/'

    def create_preferences(self):
        """Creates provider preferences."""
        from qiskit_aqua_cmd import Preferences
        return Preferences()

    def create_uipreferences(self):
        """Creates provider UI preferences."""
        return UIPreferences()

    def get_logging_level(self):
        """get level for the named logger."""
        from qiskit_chemistry._logging import get_logging_level as chem_get_logging_level
        return chem_get_logging_level()

    def set_logging_config(self, logging_config):
        """Update logger configurations using a SDK default one."""
        from qiskit_chemistry._logging import set_logging_config as chem_set_logging_config
        chem_set_logging_config(logging_config)

    def build_logging_config(self, level):
        """
         Creates a the configuration dict of the named loggers
        """
        from qiskit_chemistry._logging import build_logging_config as chem_build_logging_config
        return chem_build_logging_config(level)

    def create_section_properties_view(self, parent):
        """
        Creates provider section properties view
        """
        return ChemSectionPropertiesView(self.controller, parent)

    def add_toolbar_items(self, toolbar):
        """
        Add items to toolbar
        """
        checkButton = ttk.Checkbutton(toolbar,
                                      text="Generate Algorithm Input",
                                      variable=self.controller._save_algo_json)
        checkButton.pack(side=tk.LEFT)

    def add_file_menu_items(self, file_menu):
        """
        Add items to file menu
        """
        dict_menu = tk.Menu(file_menu, tearoff=False)
        file_menu.add_cascade(label="Export Dictionary", menu=dict_menu)
        dict_menu.add_command(label='Clipboard', command=self._export_dictionary_to_clipboard)
        dict_menu.add_command(label='File...', command=self._export_dictionary_to_file)

    def _export_dictionary_to_clipboard(self):
        if self.controller.is_empty():
            self.controller._outputView.write_line("No data to export.")
            return

        self.controller.export_dictionary_to_clipboard()

    def _export_dictionary_to_file(self):
        if self.controller.is_empty():
            self.controller._outputView.write_line("No data to export.")
            return

        preferences = self.create_uipreferences()
        filename = tkfd.asksaveasfilename(parent=self.controller.view,
                                          title='Export Chemistry Input',
                                          initialdir=preferences.get_savefile_initialdir())
        if filename and self.controller.export_dictionary_to_file(filename):
            preferences.set_savefile_initialdir(os.path.dirname(filename))
            preferences.save()
