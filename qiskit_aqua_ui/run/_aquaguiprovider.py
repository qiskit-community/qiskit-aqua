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

from qiskit_aqua_ui import GUIProvider
from ._controller import Controller
from qiskit_aqua_ui._uipreferences import UIPreferences
from ._sectionpropertiesview import SectionPropertiesView


class AquaGUIProvider(GUIProvider):
    """
    Aqua GUIProvider.
    """

    def __init__(self, controller):
        super().__init__(controller)
        self._preferences = None
        self._uipreferences = UIPreferences()

    @property
    def title(self):
        """Return provider title."""
        return 'Qiskit Aqua'

    @property
    def version(self):
        """Return provider version."""
        from qiskit_aqua import __version__
        return __version__

    @property
    def help_hyperlink(self):
        """Return provider help hyperlink."""
        return 'http://qiskit.org/documentation/aqua/'

    @property
    def preferences(self):
        """Return provider preferences."""
        from qiskit_aqua_cmd import Preferences
        if self._preferences is None:
            self._preferences = Preferences()

        return self._preferences

    @property
    def uipreferences(self):
        """Return provider UI preferences."""
        return self._uipreferences

    def get_logging_level(self):
        """get level for the named logger."""
        from qiskit_aqua._logging import get_logging_level as aqua_get_logging_level
        return aqua_get_logging_level()

    def set_logging_config(self, logging_config):
        """Update logger configurations using a SDK default one."""
        from qiskit_aqua._logging import set_logging_config as aqua_set_logging_config
        aqua_set_logging_config(logging_config)

    def build_logging_config(self, level):
        """
         Creates a the configuration dict of the named loggers
        """
        from qiskit_aqua._logging import build_logging_config as aqua_build_logging_config
        return aqua_build_logging_config(level)

    def create_section_properties_view(self, parent):
        """
        Creates provider section properties view
        """
        return SectionPropertiesView(self.controller, parent)

    def add_file_menu_items(self, file_menu):
        """
        Add items to file menu
        """
        pass
