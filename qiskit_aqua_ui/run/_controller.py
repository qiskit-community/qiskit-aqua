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

from .base_controller import BaseController
from ._model import Model
from ._customwidgets import (EntryPopup, ComboboxPopup, TextPopup)
import tkinter as tk
from tkinter import messagebox
import json
import ast
import logging

logger = logging.getLogger(__name__)


class Controller(BaseController):

    def __init__(self, guiprovider):
        super().__init__(guiprovider, Model())

    def on_section_select(self, section_name):
        self._sectionsView.show_remove_button(True)
        self._sectionView_title.set(section_name)
        if self.model.section_is_text(section_name):
            self._textView.populate(self.model.get_section_text(section_name))
            self._textView.section_name = section_name
            self._textView.show_add_button(False)
            self._textView.show_remove_button(False)
            self._textView.show_defaults_button(not self.model.default_properties_equals_properties(section_name))
            self._textView.tkraise()
        else:
            self._propertiesView.show_add_button(self.shows_add_button(section_name))
            self._propertiesView.populate(self.model.get_section_properties(section_name))
            self._propertiesView.section_name = section_name
            self._propertiesView.show_remove_button(False)
            self._propertiesView.show_defaults_button(not self.model.default_properties_equals_properties(section_name))
            self._propertiesView.tkraise()

    def on_section_defaults(self, section_name):
        try:
            self.model.set_default_properties_for_name(section_name)
            self.on_section_select(section_name)
            return True
        except Exception as e:
            messagebox.showerror("Error", str(e))

        return False

    def on_property_set(self, section_name, property_name, value):
        from qiskit.aqua.parser import JSONSchema
        try:
            self.model.set_section_property(section_name, property_name, value)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return False

        try:
            self._propertiesView.populate(self.model.get_section_properties(section_name))
            self._propertiesView.show_add_button(self.shows_add_button(section_name))
            _show_remove = property_name != JSONSchema.PROVIDER and property_name != JSONSchema.NAME \
                if section_name == JSONSchema.BACKEND else property_name != JSONSchema.NAME
            self._propertiesView.show_remove_button(_show_remove and self._propertiesView.has_selection())
            self._propertiesView.show_defaults_button(not self.model.default_properties_equals_properties(section_name))
            section_names = self.model.get_section_names()
            self._sectionsView.populate(section_names, section_name)
            missing = self.get_sections_names_missing()
            self._sectionsView.show_add_button(True if missing else False)
            return True
        except Exception as e:
            messagebox.showerror("Error", str(e))

        return False

    def on_section_property_remove(self, section_name, property_name):
        try:
            self.model.delete_section_property(section_name, property_name)
            self._propertiesView.populate(self.model.get_section_properties(section_name))
            self._propertiesView.show_add_button(self.shows_add_button(section_name))
            self._propertiesView.show_remove_button(False)
            self._propertiesView.show_defaults_button(not self.model.default_properties_equals_properties(section_name))
        except Exception as e:
            self._outputView.write_line(str(e))

    def create_popup(self, section_name, property_name, parent, value):
        from qiskit.aqua.parser import JSONSchema
        values = None
        types = ['string']
        combobox_state = 'readonly'
        if JSONSchema.NAME == property_name and Model.is_pluggable_section(section_name):
            values = self.model.get_pluggable_section_names(section_name)
        elif JSONSchema.BACKEND == section_name and \
                (JSONSchema.NAME == property_name or JSONSchema.PROVIDER == property_name):
            values = []
            if JSONSchema.PROVIDER == property_name:
                combobox_state = 'normal'
                for provider, _ in self.model.providers.items():
                    values.append(provider)
            else:
                provider_name = self.model.get_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER)
                values = self.model.providers.get(provider_name, [])
        else:
            values = self.model.get_property_default_values(section_name, property_name)
            types = self.model.get_property_types(section_name, property_name)

        if values is not None:
            widget = ComboboxPopup(self, section_name,
                                   property_name,
                                   parent,
                                   exportselection=0,
                                   state=combobox_state,
                                   values=values)
            widget._text = '' if value is None else str(value)
            if len(values) > 0:
                if value in values:
                    widget.current(values.index(value))
                else:
                    widget.current(0)

            return widget

        value = '' if value is None else value
        if 'number' in types or 'integer' in types:
            vcmd = self._validate_integer_command if 'integer' in types else self._validate_float_command
            vcmd = (vcmd, '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
            widget = EntryPopup(self,
                                section_name,
                                property_name,
                                parent,
                                value,
                                validate='all',
                                validatecommand=vcmd,
                                state=tk.NORMAL)
            widget.selectAll()
            return widget

        if 'object' in types or 'array' in types:
            try:
                if isinstance(value, str):
                    value = value.strip()
                    if len(value) > 0:
                        value = ast.literal_eval(value)

                if isinstance(value, dict) or isinstance(value, list):
                    value = json.dumps(value, sort_keys=True, indent=4)
            except:
                pass

        widget = TextPopup(self,
                           section_name,
                           property_name,
                           parent,
                           value)
        widget.selectAll()
        return widget
