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

class SectionPropertiesView(ScrollbarView):

    def __init__(self, controller, parent, **options):
        super(SectionPropertiesView, self).__init__(parent, **options)
        self._controller = controller
        self._tree = None
        
    def clear(self):
        if self._tree is not None:
            for i in self._tree.get_children():
                self._tree.delete([i])
            
    def populate(self,column_titles,properties):
        self.clear()
        ttk.Style().configure("BrowseSectionPropertiesView.Treeview.Heading", font=(None,12,'bold')) 
        self._tree = ttk.Treeview(self,style='BrowseSectionPropertiesView.Treeview', selectmode=tk.BROWSE, columns=column_titles)
        self._tree.heading('#0', text='property')
        self.init_widgets(self._tree)
        for value in column_titles:
            self._tree.heading(value,text=value)
            
        self._controller._propertiesView.grid(row=0,column=0,sticky='nsew')
            
        for name,props in properties.items():
            values = [''] * len(column_titles)
            for k,v in props.items():
                index = column_titles.index(k)
                if isinstance(v,list) and len(v) == 0:
                    v = str(v)
                values[index] = ','.join(str(t) for t in v) if isinstance(v,list) else str(v)
            
            self._tree.insert('',tk.END, text=name, values=values)
            
