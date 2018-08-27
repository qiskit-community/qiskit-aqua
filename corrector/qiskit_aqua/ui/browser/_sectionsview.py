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
import tkinter.ttk as ttk
from qiskit_aqua.ui.browser._scrollbarview import ScrollbarView

class SectionsView(ScrollbarView):

    def __init__(self, controller, parent, **options):
        super(SectionsView, self).__init__(parent, **options)
        self._controller = controller
        ttk.Style().configure("BrowseSectionsView.Treeview.Heading", font=(None,12,'bold'))
        self._tree = ttk.Treeview(self,style='BrowseSectionsView.Treeview', selectmode=tk.BROWSE)
        self._tree.heading('#0', text='Sections')
        self._tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        self.init_widgets(self._tree)
        
    def clear(self):
        for i in self._tree.get_children():
            self._tree.delete([i])
            
    def populate(self,algos):
        self.clear()
        root_identifier = None
        for main_name,sections in algos.items():
            identifier = self._tree.insert('',
                                           tk.END, 
                                           text=main_name,
                                           values=[''])
            if root_identifier is None:
                root_identifier = identifier
                
            child_identifier = None
            for algo_name,_ in sections.items():
                child_identifier = self._tree.insert(identifier,
                                                     tk.END, 
                                                     text=algo_name, 
                                                     values=[main_name])
                
            if child_identifier is not None:
                self._tree.see(child_identifier)
                
        if root_identifier is not None:
                self._tree.see(root_identifier)
     
    def has_selection(self):
        return self._tree.selection()
    
    def _on_tree_select(self,event):
        for item in self._tree.selection():
            item_text = self._tree.item(item,'text')
            if item_text in self._controller.top_names():
                self._controller.on_top_name_select(item_text)
            else:
                values = self._tree.item(item,'values')
                self._controller.on_algo_select(values[0],item_text)
            return
