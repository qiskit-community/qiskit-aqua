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
from qiskit_aqua.ui.run._scrollbarview import ScrollbarView
from qiskit_aqua.ui.run._customwidgets import TextCustom
import queue
import string

class ThreadSafeOutputView(ScrollbarView):
    
    _DELAY = 50
    
    def __init__(self,parent,**options):
        super(ThreadSafeOutputView, self).__init__(parent, **options)
        self._queue = queue.Queue()
        self._textWidget = TextCustom(self,wrap=tk.NONE,state=tk.DISABLED)
        self.init_widgets(self._textWidget)
        self._updateText()
        
    def _updateText(self):
        try:
            iterations = 0
            while iterations < 120:
                line = self._queue.get_nowait()
                iterations += 1
                if line is None:
                    self._write()
                else:
                    self._write(str(line),False)
                self.update_idletasks()
        except:
            pass
        
        self.after(ThreadSafeOutputView._DELAY,self._updateText)
    
    def write(self,text):
        if text is not None:
            text = str(text)
            if len(text) > 0:
                # remove any non printable character that will cause the Text widgetto hang
                text = ''.join([x if x in string.printable else '' for x in text])
                if len(text) > 0: 
                    self._queue.put(text)

    def flush(self):
        pass
    
    def buffer_empty(self):
        return self._queue.empty()
    
    def clear_buffer(self):
        """
        Create another queue to ignore current queue output
        """
        self._queue = queue.Queue()
        
    def write_line(self,text):
        self.write(text + '\n')
        
    def clear(self):
        self._queue.put(None)
    
    def _write(self,text=None,erase=True):
        self._textWidget.config(state=tk.NORMAL)
        if erase:
            self._textWidget.delete(1.0, tk.END)
            
        if text is not None:
            self._textWidget.insert(tk.END, text)
            pos = self._vscrollbar.get()[1]
            # scrolls only when scroll bar is at the bottom
            if pos == 1.0:
                self._textWidget.yview(tk.END)
            
        self._textWidget.config(state=tk.DISABLED)
