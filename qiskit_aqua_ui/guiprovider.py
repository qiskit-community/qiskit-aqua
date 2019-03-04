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

from abc import ABC, abstractmethod


class GUIProvider(ABC):
    """
    Base class for GUIProviders.
    """

    START, STOP = 'Start', 'Stop'

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def title(self):
        """Return provider title."""
        pass

    @property
    @abstractmethod
    def version(self):
        """Return provider version."""
        pass

    @property
    @abstractmethod
    def help_hyperlink(self):
        """Return provider help hyperlink."""
        pass

    @property
    @abstractmethod
    def controller(self):
        """Return provider controller."""
        pass

    @abstractmethod
    def create_uipreferences(self):
        """Creates provider UI preferences."""
        pass

    @abstractmethod
    def get_logging_level(self):
        """get level for the named logger."""
        pass

    @abstractmethod
    def set_logging_config(self, logging_config):
        """Update logger configurations using a SDK default one."""
        pass

    @abstractmethod
    def build_logging_config(self, level):
        """
         Creates a the configuration dict of the named loggers
        """
        pass

    @abstractmethod
    def create_section_properties_view(self, parent):
        """
        Creates provider section properties view
        """
        pass

    @abstractmethod
    def add_toolbar_items(self, toolbar):
        """
        Add items to toolbar
        """
        pass

    @abstractmethod
    def add_file_menu_items(self, file_menu):
        """
        Add items to file menu
        """
        pass

    @abstractmethod
    def create_run_thread(self, model, outputview, thread_queue):
        """
        Creates run thread
        """
        pass
