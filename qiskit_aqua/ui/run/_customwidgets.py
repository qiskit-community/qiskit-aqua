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

from sys import platform
import tkinter as tk
import tkinter.ttk as ttk
from qiskit_aqua.ui.run._dialog import Dialog

_BIND = '<Button-2><ButtonRelease-2>' if platform == 'darwin' else '<Button-3><ButtonRelease-3>'
_LINESEP = '\n'

class EntryCustom(ttk.Entry):
    
    def __init__(self, *args, **kwargs):
        super(EntryCustom, self).__init__(*args, **kwargs)
        _create_menu(self)
        self.bind('<Button-1><ButtonRelease-1>', self._dismiss_menu)
        self.bind_class('Entry', '<Control-a>', self._event_select_all)  
        self.bind(_BIND, self._show_menu)
        self.bind('<<Paste>>',self._event_paste)

    def _event_select_all(self, *args):
        if platform == 'darwin':
            self.focus_force()
        self.selection_range(0, tk.END)
        return 'break'

    def _show_menu(self, e):
        self.menu.post(e.x_root, e.y_root)
        if platform == 'darwin':
            self.selection_clear()
        
    def _dismiss_menu(self, e):
        self.menu.unpost()
        
    def _event_paste(self,e):
        try:
            self.delete(tk.SEL_FIRST,tk.SEL_LAST)
        except:
            pass
        
        try:
            self.insert(tk.INSERT, self.clipboard_get())
        except:
            pass
        
        return 'break'

class TextCustom(tk.Text):
    
    def __init__(self, *args, **kwargs):
        super(TextCustom, self).__init__(*args, **kwargs)
        _create_menu(self)
        self.bind('<Button-1><ButtonRelease-1>', self._dismiss_menu)
        self.bind_class('Text', '<Control-a>', self._event_select_all)  
        self.bind(_BIND, self._show_menu)
        self.bind('<1>', lambda event: self.focus_set())
        self.bind('<<Paste>>',self._event_paste)
      
    def _event_select_all(self, *args):
        # do not select the new line that the text widget automatically adds at the end
        self.tag_add(tk.SEL,1.0,tk.END + '-1c')
        return 'break'

    def _show_menu(self, e):
        self.menu.post(e.x_root, e.y_root)
        
    def _dismiss_menu(self, e):
        self.menu.unpost()
        
    def _event_paste(self,e):
        try:
            self.delete(tk.SEL_FIRST,tk.SEL_LAST)
        except:
            pass
        
        try:
            self.insert(tk.INSERT, self.clipboard_get())
        except:
            pass
        
        return 'break'
        
class EntryPopup(EntryCustom):

    def __init__(self, controller,section_name,property_name,parent, text, **options):
        ''' If relwidth is set, then width is ignored '''
        super(EntryPopup, self).__init__(parent,**options)
        self._controller = controller
        self._section_name = section_name
        self._property_name = property_name
        self._text = text
        self.insert(0, self._text) 
        self.focus_force()
        self.bind("<Unmap>", self._update_value)
        self.bind("<FocusOut>", self._update_value)
        
    def selectAll(self):
        self.focus_force()
        self.selection_range(0, tk.END)
    
    def _update_value(self, *ignore):
        new_text = self.get()
        valid = True
        if self._text != new_text:
            self._text = new_text
            valid = self._controller.on_property_set(self._section_name,
                                                          self._property_name,
                                                          new_text)
        if valid:
            self.destroy()
        else:
            self.selectAll() 
        
class ComboboxPopup(ttk.Combobox):

    def __init__(self, controller,section_name,property_name,parent, **options):
        ''' If relwidth is set, then width is ignored '''
        super(ComboboxPopup, self).__init__(parent,**options)
        self._controller = controller
        self._section_name = section_name
        self._property_name = property_name
        self.focus_force()
        self.bind("<Unmap>", self._update_value)
        self.bind("<FocusOut>", self._update_value)
        self.bind("<<ComboboxSelected>>", self._on_select)
        self._text = None
        
    def _on_select(self,  *ignore):
        new_text = self.get()
        if len(new_text) > 0 and self._text != new_text:
            self._text = new_text
            self._controller.on_property_set(self._section_name,
                                             self._property_name,
                                             new_text)
            
    def _update_value(self,  *ignore):
        new_text = self.get()
        state = self.state()
        if isinstance(state,tuple) and state[0] != 'pressed':
            self.destroy()
            
        if len(new_text) > 0 and self._text != new_text:
            self._text = new_text
            self._controller.on_property_set(self._section_name,
                                             self._property_name,
                                             new_text)
            
class TextPopup(ttk.Frame):

    def __init__(self, controller,section_name,property_name,parent, text, **options):
        super(TextPopup, self).__init__(parent,**options)
        self._child = TextCustom(self,wrap=tk.NONE,state=tk.NORMAL)
        self._hscrollbar = ttk.Scrollbar(self, orient = tk.HORIZONTAL)
        self._vscrollbar = ttk.Scrollbar(self, orient = tk.VERTICAL)
        self._child.config(yscrollcommand = self._vscrollbar.set)
        self._child.config(xscrollcommand = self._hscrollbar.set)
        self._vscrollbar.config(command = self._child.yview)
        self._hscrollbar.config(command = self._child.xview)
        
        
        self._hscrollbar.pack(side=tk.BOTTOM, fill=tk.X, expand=tk.FALSE)
        self._vscrollbar.pack(side=tk.RIGHT, fill=tk.Y, expand=tk.FALSE)
        self._child.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)
        self.pack()
        
        self._controller = controller
        self._section_name = section_name
        self._property_name = property_name
        self._text = text
        if self._text is not None:
            self._child.insert(tk.END, self._text)
             
        self._child.focus_force()
        self.bind("<Unmap>", self._update_value)
        self.bind("<FocusOut>", self._update_value)
        
    def selectAll(self):
        self._child.focus_force()
        # do not select the new line that the text widget automatically adds at the end
        self._child.tag_add(tk.SEL,1.0,tk.END + '-1c')
        
    def _update_value(self, *ignore):
        sep_pos = -len(_LINESEP)
        new_text = self._child.get(1.0, tk.END)
        if len(new_text) >= len(_LINESEP) and new_text[sep_pos:] == _LINESEP:
            new_text = new_text[:sep_pos]
               
        valid = True
        if self._text != new_text:
            self._text = new_text
            valid = self._controller.on_property_set(self._section_name,
                                                     self._property_name,
                                                     new_text)
        if valid:
            self.destroy()
        else:
            self.selectAll() 
            
class PropertyEntryDialog(Dialog):
    
    def __init__(self,controller,section_name,parent):
        super(PropertyEntryDialog, self).__init__(controller,parent,"New Property")
        self._section_name = section_name
        self.label_text = None
        self.label = None
       
    def body(self, parent,options):
        ttk.Label(parent,
                  text="Name:",
                  borderwidth=0).grid(padx=7,pady=6,row=0)
        
        self.entry = EntryCustom(parent,state=tk.NORMAL)
        self.entry.grid(padx=(0,7),pady=6,row=0, column=1)
        self.label_text = tk.StringVar()
        self.label = ttk.Label(parent,foreground='red',
                               textvariable=self.label_text,
                               borderwidth=0)
        self.label.grid(padx=(7,7),
                        pady=6,
                        row=1,
                        column=0,
                        columnspan=2)
        self.label.grid_remove()
        return self.entry # initial focus
    
    def validate(self):
        self.label.grid_remove()
        self.label_text = self.controller.validate_property_add(self._section_name,
                                                    self.entry.get().strip())
        if self.label_text is None:
            return True
        
        self.label.grid()
        return False

    def apply(self):
        self.result = self.entry.get()

class PropertyComboDialog(Dialog):
    
    def __init__(self,controller,section_name,parent):
        super(PropertyComboDialog, self).__init__(controller,parent,'New Property')
        self._section_name = section_name
        self.label_text = None
        self.label = None
       
    def body(self, parent,options):
        ttk.Label(parent,
                  text="Name:",
                  borderwidth=0).grid(padx=7,pady=6,row=0)
        self.entry = ttk.Combobox(parent,
                                  exportselection=0,
                                  state='readonly',
                                  values=options['values'])
        self.entry.current(0)
        self.entry.grid(padx=(0,7),pady=6,row=0, column=1)
        self.label_text = tk.StringVar()
        self.label = ttk.Label(parent,foreground='red',
                               textvariable=self.label_text,
                               borderwidth=0)
        self.label.grid(padx=(7,7),
                        pady=6,
                        row=1,
                        column=0,
                        columnspan=2)
        self.label.grid_remove()
        return self.entry # initial focus
    
    def validate(self):
        self.label.grid_remove()
        self.label_text = self.controller.validate_property_add(self._section_name,
                                                    self.entry.get().strip())
        if self.label_text is None:
            return True
        
        self.label.grid()
        return False

    def apply(self):
        self.result = self.entry.get()
        
class SectionComboDialog(Dialog):
    
    def __init__(self,controller,parent):
        super(SectionComboDialog, self).__init__(controller,parent,"New Section")
        self.label_text = None
        self.label = None
       
    def body(self, parent,options):
        ttk.Label(parent,
                  text='Name:',
                  borderwidth=0).grid(padx=7,
                                       pady=6,
                                       row=0)
        self.entry = ttk.Combobox(parent,
                                  exportselection=0,
                                  state='readonly',
                                  values=options['sections'])
        self.entry.current(0)
        self.entry.grid(padx=(0,7),pady=6,row=0, column=1)
        self.label_text = tk.StringVar()
        self.label = ttk.Label(parent,foreground='red',
                               textvariable=self.label_text,
                               borderwidth=0)
        self.label.grid(padx=(7,7),
                        pady=6,
                        row=1,
                        column=0,
                        columnspan=2)
        self.label.grid_remove()
        return self.entry # initial focus
    
    def validate(self):
        self.label.grid_remove()
        self.label_text = self.controller.validate_section_add(self.entry.get().lower().strip())
        if self.label_text is None:
            return True
        
        self.label.grid()
        return False

    def apply(self):
        self.result = self.entry.get().lower().strip()
        
def _create_menu(w):
    state = str(w['state'])
    w.menu = tk.Menu(w, tearoff=0)
    if state == tk.NORMAL:
        w.menu.add_command(label='Cut')
    w.menu.add_command(label='Copy')
    if state == tk.NORMAL:
        w.menu.add_command(label='Paste')
    w.menu.add_separator()
    w.menu.add_command(label='Select all')        

    if state == tk.NORMAL:
        w.menu.entryconfigure('Cut', 
                              command=lambda: w.focus_force() or w.event_generate('<<Cut>>'))
    w.menu.entryconfigure('Copy', 
                          command=lambda: w.focus_force() or w.event_generate('<<Copy>>'))
    if state == tk.NORMAL:
        w.menu.entryconfigure('Paste', 
                              command=lambda: w.focus_force() or w.event_generate('<<Paste>>'))
        
    if platform == 'darwin' and isinstance(w,ttk.Entry):
        w.menu.entryconfigure('Select all', 
                              command=lambda: w.after(0, w._event_select_all))  
    else:
        w.menu.entryconfigure('Select all',
                              command=lambda: w.focus_force() or w._event_select_all(None))