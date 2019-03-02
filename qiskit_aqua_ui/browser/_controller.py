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

from ._model import Model
import tkinter as tk
import logging

logger = logging.getLogger(__name__)


class Controller(object):

    _NAME = 'name'

    def __init__(self, view):
        self._view = view
        self._model = None
        self._filemenu = None
        self._sectionsView = None
        self._emptyView = None
        self._sectionsView_title = tk.StringVar()
        self._propertiesView = None

    @property
    def model(self):
        if self._model is None:
            self._model = Model()

        return self._model

    def pluggable_names(self):
        return self.model.pluggable_names()

    def get_property_titles(self, section_name):
        return self.model.get_property_titles(section_name)

    def populate_sections(self):
        self._sectionsView.populate(self.model.get_sections())

    def on_pluggable_type_select(self, pluggable_type):
        self._sectionsView_title.set('')
        self._emptyView.tkraise()

    def on_pluggable_schema_select(self, pluggable_type, pluggable_name):
        self._sectionsView_title.set(self.model.get_pluggable_description(pluggable_type, pluggable_name))
        properties = self.model.get_pluggable_schema_properties(pluggable_type, pluggable_name)
        column_titles = self.model.get_pluggable_schema_property_titles(pluggable_type, pluggable_name)
        self._propertiesView.populate(column_titles, properties)
        self._propertiesView.tkraise()

    def on_pluggable_problems_select(self, pluggable_type, pluggable_name):
        problems = self.model.get_pluggable_problems(pluggable_type, pluggable_name)
        self._propertiesView.populate(['oneOf'], {'name': {'oneOf': problems}})
        self._propertiesView.tkraise()

    def on_pluggable_depends_select(self, pluggable_type, pluggable_name):
        self._sectionsView_title.set(self.model.get_pluggable_description(pluggable_type, pluggable_name))
        self._emptyView.tkraise()

    def on_pluggable_dependency_select(self, pluggable_type, pluggable_name, dependency_type):
        dependency = self.model.get_pluggable_dependency(pluggable_type, pluggable_name, dependency_type)
        default = dependency.get('default', {})
        self._propertiesView.populate(list(default.keys()), {'default': default})
        self._propertiesView.tkraise()
