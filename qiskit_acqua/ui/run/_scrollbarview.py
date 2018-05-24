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

class ScrollbarView(ttk.Frame):

    def __init__(self, parent,**options):
        super(ScrollbarView, self).__init__(parent, **options)
        self._child = None
        self._hscrollbar = None
        self._vscrollbar = None
        
    def init_widgets(self, child):     
        self._child = child
        self._hscrollbar = ttk.Scrollbar(self, orient = tk.HORIZONTAL)
        self._vscrollbar = ttk.Scrollbar(self, orient = tk.VERTICAL)
        self._child.config(yscrollcommand = self._vscrollbar.set)
        self._child.config(xscrollcommand = self._hscrollbar.set)
        self._vscrollbar.config(command = self._child.yview)
        self._hscrollbar.config(command = self._child.xview)
        
    def pack(self, **options):
        if self._hscrollbar is not None: 
            self._hscrollbar.pack(side=tk.BOTTOM, fill=tk.X, expand=tk.FALSE)
        
        if self._vscrollbar is not None:
            self._vscrollbar.pack(side=tk.RIGHT, fill=tk.Y, expand=tk.FALSE)
        
        if self._child is not None:
            self._child.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)
        
        ttk.Frame.pack(self, **options)
        
    def grid(self, **options):
        if self._hscrollbar is not None: 
            self._hscrollbar.pack(side=tk.BOTTOM, fill=tk.X, expand=tk.FALSE)
            
        if self._vscrollbar is not None:
            self._vscrollbar.pack(side=tk.RIGHT, fill=tk.Y, expand=tk.FALSE)
        
        if self._child is not None:
            self._child.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)
        
        ttk.Frame.grid(self, **options)
