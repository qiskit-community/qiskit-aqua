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
from qiskit_aqua.ui.run._scrollbarview import ScrollbarView

class ToolbarView(ScrollbarView):

    def __init__(self, parent,**options):
        super(ScrollbarView, self).__init__(parent, **options)
        self._child = None
        self._toolbar = None
        self._add_button = None
        self._remove_button = None
        self._defaults_button = None
        self._hscrollbar = None
        self._vscrollbar = None
        self._add_button_shown = False
        self._remove_button_shown = False
        self._defaults_button_shown = False
        self._makeToolBar()
        
    def _makeToolBar(self):
        self._toolbar = ttk.Frame(self)
        self._add_button = ttk.Button(self._toolbar,
                                       text='Add',
                                       state='enable',
                                       command=self.onadd)
        self._remove_button = ttk.Button(self._toolbar,
                                       text='Remove',
                                       state='enable',
                                       command=self.onremove)
        self._defaults_button = ttk.Button(self._toolbar,
                                       text='Defaults',
                                       state='enable',
                                       command=self.ondefaults)
        
    def onadd(self):
        pass
            
    def onremove(self):
        pass
    
    def ondefaults(self):
        pass
    
    def get_toolbar_size(self):
        if self._toolbar is None:
            return (0,0)
        
        return (self._toolbar.winfo_width(),self._toolbar.winfo_height())
    
    def pack(self, **options):
        if self._toolbar is not None:
            self._toolbar.pack(side=tk.BOTTOM,fill=tk.X)
            self._add_button.pack(side=tk.LEFT)
            self._remove_button.pack(side=tk.LEFT)
            self._defaults_button.pack(side=tk.RIGHT)
            
        ScrollbarView.pack(self,**options)
        
    def grid(self, **options):
        if self._toolbar is not None:
            self._toolbar.pack(side=tk.BOTTOM,fill=tk.X)
            self._add_button.pack(side=tk.LEFT)
            self._remove_button.pack(side=tk.LEFT)
            self._defaults_button.pack(side=tk.RIGHT)
            
        ScrollbarView.grid(self,**options)
        
    def show_add_button(self,show):
        self._add_button_shown = show
        if show:
            if self._remove_button_shown:
                self._remove_button.pack_forget()
            self._add_button.pack(side=tk.LEFT)
            if self._remove_button_shown:
                self._remove_button.pack(side=tk.LEFT)
        else:
            self._add_button.pack_forget()
            
    def show_remove_button(self,show):
        self._remove_button_shown = show
        if show:
            self._remove_button.pack(side=tk.LEFT)
        else:
            self._remove_button.pack_forget()
            
    def show_defaults_button(self,show):
        self._defaults_button_shown = show
        if show:
            self._defaults_button.pack(side=tk.RIGHT)
        else:
            self._defaults_button.pack_forget()
       
