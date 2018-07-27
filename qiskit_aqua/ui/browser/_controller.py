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

from qiskit_aqua.ui.browser._model import Model
import tkinter as tk
import logging

logger = logging.getLogger(__name__)

class Controller(object):
     
    _NAME = 'name'
    
    def __init__(self,view):
        self._view = view
        self._model = Model()
        self._filemenu = None
        self._sectionsView = None
        self._emptyView = None
        self._sectionsView_title = tk.StringVar()
        self._propertiesView = None
        
    def top_names(self):
        return self._model.top_names()
    
    def get_property_titles(self,section_name):
        return self._model.get_property_titles(section_name)
        
    def populate_sections(self):
        self._sectionsView.populate(self._model.get_sections())
    
    def on_top_name_select(self,top_name):
        self._sectionsView_title.set('')
        self._emptyView.tkraise()
        
    def on_algo_select(self,top_name,section_name): 
        self._sectionsView_title.set(self._model.get_section_description(top_name,section_name))
        properties = self._model.get_section_properties(top_name,section_name)
        column_titles = self._model.get_property_titles(top_name,section_name)
        self._propertiesView.populate(column_titles,properties)
        self._propertiesView.tkraise()

        
                                  
        
        
