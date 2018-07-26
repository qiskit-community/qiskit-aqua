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

import tkinter as tk
from qiskit_aqua.ui.run._toolbarview import ToolbarView
from qiskit_aqua.ui.run._customwidgets import TextCustom

_LINESEP = '\n'

class SectionTextView(ToolbarView):

    def __init__(self, controller, parent, **options):
        super(SectionTextView, self).__init__(parent, **options)
        self._controller = controller
        self._textWidget = TextCustom(self,wrap=tk.NONE,state=tk.NORMAL)
        self.init_widgets(self._textWidget)
        self.bind("<Unmap>", self._update_value)
        self.bind("<FocusOut>", self._update_value)
        self._section_name = None
        self._text = None
        
    @property
    def section_name(self):
        return self._section_name

    @section_name.setter
    def section_name(self, new_section_name):
        self._section_name = new_section_name
        
    def populate(self,text):
        self._textWidget.delete(1.0, tk.END)
        if text is not None:
            self._textWidget.insert(tk.END, text)
        
        self._text = text
        
    def clear(self):
        self._textWidget.delete(1.0, tk.END)
        self._text = self._textWidget.get(1.0, tk.END)
            
    def _update_value(self, *ignore):
        sep_pos = -len(_LINESEP)
        new_text = self._textWidget.get(1.0, tk.END)
        if len(new_text) >= len(_LINESEP) and new_text[sep_pos:] == _LINESEP:
            new_text = new_text[:sep_pos]
                    
        if self._text != new_text:
            self._text = new_text
            self._controller.on_text_set(self._section_name,new_text)
            
    def ondefaults(self):
        self._controller.on_section_defaults(self.section_name)
