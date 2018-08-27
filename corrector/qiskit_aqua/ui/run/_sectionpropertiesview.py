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
from qiskit_aqua.ui.run._toolbarview import ToolbarView
from qiskit_aqua.ui.run._customwidgets import PropertyComboDialog,PropertyEntryDialog,TextPopup

class SectionPropertiesView(ToolbarView):

    def __init__(self, controller, parent, **options):
        super(SectionPropertiesView, self).__init__(parent, **options)
        self._controller = controller
        ttk.Style().configure("SectionPropertiesView.Treeview.Heading", font=(None,12,'bold'))
        self._tree = ttk.Treeview(self,style='SectionPropertiesView.Treeview', selectmode=tk.BROWSE,columns=['value'])
        self._tree.heading('#0', text='Name')
        self._tree.heading('value',text='Value')
        self._tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        self._tree.bind('<Button-1>', self._on_tree_edit)
        self.init_widgets(self._tree)
        self._section_name = None
        self._properties = {}
        self._popup_widget = None
        
    @property
    def section_name(self):
        return self._section_name

    @section_name.setter
    def section_name(self, new_section_name):
        self._section_name = new_section_name
        
    def clear(self):
        if self._popup_widget is not None and self._popup_widget.winfo_exists():
            self._popup_widget.destroy()
            
        self._popup_widget = None
        for i in self._tree.get_children():
            self._tree.delete([i])
            
        self._properties = {}
            
    def populate(self,properties):
        self.clear()
        for name,value in properties.items():
            value = '' if value is None else str(value)
            value = value.replace('\r', '\\r').replace('\n', '\\n')
            self._tree.insert('',tk.END, text=name, values=[value])
            
        self._properties = properties
            
    def set_property(self,property_name,value):
        for item in self._tree.get_children():
            name = self._tree.item(item, "text")
            if name == property_name:
                self._tree.item(item, values=[value])
                break
            
    def has_selection(self):
        return self._tree.selection()
            
    def _on_tree_select(self,event):
        for item in self._tree.selection():
            property_name = self._tree.item(item,'text')
            self._controller.on_property_select(self._section_name,property_name)
            return
        
    def _on_tree_edit(self,event):
        rowid = self._tree.identify_row(event.y)
        if not rowid:
            return
    
        column = self._tree.identify_column(event.x)
        if column == '#1':
            x,y,width,height = self._tree.bbox(rowid, column)
            pady = height // 2
           
            item = self._tree.identify("item", event.x, event.y)
            property_name = self._tree.item(item, "text")
            self._popup_widget = self._controller.create_popup(self.section_name,
                                                   property_name,
                                                   self._tree,
                                                   self._properties[property_name])
            if isinstance(self._popup_widget,TextPopup):
                height = self._tree.winfo_height() - y
                self._popup_widget.place(x=x, y=y, width=width, height=height)
            else:
                self._popup_widget.place(x=x, y=y+pady, anchor=tk.W, width=width)
        
    def onadd(self):
        dialog = None
        if self._controller._model.allows_additional_properties(self.section_name):
            dialog = PropertyEntryDialog(self._controller,self.section_name,self.master)
            dialog.do_init()
        else:
            properties = self._controller.get_property_names_missing(self.section_name)
            dialog = PropertyComboDialog(self._controller,self.section_name,self.master)
            dialog.do_init(values=properties)
         
        dialog.do_modal()
        if dialog.result is None:
            return
        
        if dialog.result is not None and len(dialog.result) > 0:
            self._controller.on_property_add(self.section_name,dialog.result)
            
    def onremove(self):
        for item in self._tree.selection():
            property_name = self._tree.item(item,'text')
            self._controller.on_section_property_remove(self.section_name,property_name)
            break
        
    def ondefaults(self):
        self._controller.on_section_defaults(self.section_name)
