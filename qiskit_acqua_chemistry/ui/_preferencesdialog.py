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
from tkinter import font
from qiskit_acqua_chemistry.ui._dialog import Dialog
from collections import OrderedDict
from qiskit_acqua_chemistry.core import refresh_operators
from qiskit_acqua_chemistry.drivers import ConfigurationManager
from qiskit_acqua_chemistry.ui._qconfigview import QconfigView
from qiskit_acqua_chemistry.ui._toolbarview import ToolbarView
from qiskit_acqua_chemistry.ui._customwidgets import EntryCustom
from qiskit_acqua_chemistry.preferences import Preferences
from qiskit_acqua_chemistry.ui._uipreferences import UIPreferences
from qiskit_acqua_chemistry._logging import get_logger_levels_for_names,build_logging_config,set_logger_config
import logging

class PreferencesDialog(Dialog):
    
    _LOG_LEVELS = OrderedDict(
            [(logging.CRITICAL , logging.getLevelName(logging.CRITICAL)),
             (logging.ERROR , logging.getLevelName(logging.ERROR)),
             (logging.WARNING , logging.getLevelName(logging.WARNING)),
             (logging.INFO , logging.getLevelName(logging.INFO)),
             (logging.DEBUG , logging.getLevelName(logging.DEBUG)),
             (logging.NOTSET , logging.getLevelName(logging.NOTSET))]
    )
    
    def __init__(self,controller,parent):
        super(PreferencesDialog, self).__init__(controller,parent,'Preferences')
        self._qconfigview = None
        self._levelCombo = None
        self._checkButton = None
        self._packagesPage = None
        self._populateDefaults = tk.IntVar()
       
    def body(self,parent,options):
        preferences = Preferences()
        logging_config = preferences.get_logging_config()
        if logging_config is not None:
            set_logger_config(logging_config)
        
        uipreferences = UIPreferences()
        populate = uipreferences.get_populate_defaults(True)
        self._populateDefaults.set(1 if populate else 0)
        
        qiskitGroup = ttk.LabelFrame(parent,
                                     text='QISKit Configuration',
                                     padding=(6,6,6,6),
                                     borderwidth=4,
                                     relief=tk.GROOVE)
        qiskitGroup.grid(padx=(7,7),pady=6,row=0, column=0,sticky='nsew')
        self._qconfigview = QconfigView(qiskitGroup)
       
        defaultsGroup = ttk.LabelFrame(parent,
                                     text='Defaults',
                                     padding=(6,6,6,6),
                                     borderwidth=4,
                                     relief=tk.GROOVE)
        defaultsGroup.grid(padx=(7,7),pady=6,row=1, column=0,sticky='nsw')
        defaultsGroup.columnconfigure(1,pad=7)
        
        self._checkButton = ttk.Checkbutton(defaultsGroup,
                                            text="Populate on file new/open",
                                            variable=self._populateDefaults)
        self._checkButton.grid(row=0, column=1,sticky='nsw')
        
        packagesGroup = ttk.LabelFrame(parent,
                                     text='Packages',
                                     padding=(6,6,6,6),
                                     borderwidth=4,
                                     relief=tk.GROOVE)
        packagesGroup.grid(padx=(7,7),pady=6,row=2, column=0,sticky='nsw')
        packagesGroup.columnconfigure(1,pad=7)
        
        frame = ttk.Frame(packagesGroup)
        frame.grid(row=0, column=0,sticky='nsew')
        
        self._packagesPage = PackagesPage(frame,preferences)
        self._packagesPage.pack(side=tk.TOP,fill=tk.BOTH, expand=tk.TRUE)
        self._packagesPage.show_add_button(True)
        self._packagesPage.show_remove_button(self._packagesPage.has_selection())
        self._packagesPage.show_defaults_button(False)
        
        loggingGroup = ttk.LabelFrame(parent,
                                     text='Logging Configuration',
                                     padding=(6,6,6,6),
                                     borderwidth=4,
                                     relief=tk.GROOVE)
        loggingGroup.grid(padx=(7,7),pady=6,row=3, column=0,sticky='nsw')
        loggingGroup.columnconfigure(1,pad=7)
        
        levels = get_logger_levels_for_names(['qiskit_acqua_chemistry','qiskit_acqua'])
        loglevel = levels[0]
        
        ttk.Label(loggingGroup,
                  text="Level:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=0, column=0,sticky='nsew')
        self._levelCombo = ttk.Combobox(loggingGroup,
                                  exportselection=0,
                                  state='readonly',
                                  values=list(PreferencesDialog._LOG_LEVELS.values()))
        index = list(PreferencesDialog._LOG_LEVELS.keys()).index(loglevel)
        self._levelCombo.current(index)
        self._levelCombo.grid(row=0, column=1,sticky='nsw')
        
        self.entry = self._qconfigview.initial_focus
        return self.entry # initial focus
    
    def validate(self):
        if not self._qconfigview.validate():
            self.initial_focus = self._qconfigview.initial_focus
            return False
        
        if not self._packagesPage.validate():
            self.initial_focus = self._packagesPage.initial_focus
            return False
        
        self.initial_focus = self._qconfigview.initial_focus
        return True

    def apply(self):
        try:
            level_name = self._levelCombo.get()
            levels = [key for key, value in PreferencesDialog._LOG_LEVELS.items() if value == level_name]
            loglevel = levels[0]
            logging_config = build_logging_config(['qiskit_acqua_chemistry','qiskit_acqua'],loglevel)
        
            preferences = Preferences()
            self._qconfigview.apply(preferences)
            self._packagesPage.apply(preferences)
            preferences.set_logging_config(logging_config)
            preferences.save()
            set_logger_config(logging_config)
        
            uipreferences = UIPreferences()
            populate = self._populateDefaults.get()
            uipreferences.set_populate_defaults(False if populate == 0 else True)
            uipreferences.save()
        
            self._controller.get_available_backends()
        except Exception as e:
            self.controller.outputview.write_line(str(e))

class PackagesPage(ToolbarView):

    def __init__(self, parent, preferences, **options):
        super(PackagesPage, self).__init__(parent, **options)
        size = font.nametofont('TkHeadingFont').actual('size')
        ttk.Style().configure("PackagesPage.Treeview.Heading", font=(None,size,'bold'))
        self._tree = ttk.Treeview(self, style='PackagesPage.Treeview', selectmode=tk.BROWSE, height=4,columns=['value'])
        self._tree.heading('#0', text='Type')
        self._tree.heading('value',text='Name')
        self._tree.column('#0',minwidth=0,width=150,stretch=tk.NO)
        self._tree.column('value',minwidth=0,width=500,stretch=tk.YES)
        self._tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        self._tree.bind('<Button-1>', self._on_tree_edit)
        self.init_widgets(self._tree)
        
        self._preferences = Preferences()
        self._popup_widget = None
        self.pack(fill=tk.BOTH, expand=tk.TRUE)
        self.populate()
        self.initial_focus = self._tree
        
    def clear(self):
        if self._popup_widget is not None and self._popup_widget.winfo_exists():
            self._popup_widget.destroy()
            
        self._popup_widget = None
        for i in self._tree.get_children():
            self._tree.delete([i])
            
    def populate(self):
        self.clear()
        packages = self._preferences.get_packages(Preferences.PACKAGE_TYPE_DRIVERS,[])
        for package in packages:
            self._populate(Preferences.PACKAGE_TYPE_DRIVERS,package)
            
        packages = self._preferences.get_packages(Preferences.PACKAGE_TYPE_CHEMISTRY,[])
        for package in packages:
            self._populate(Preferences.PACKAGE_TYPE_CHEMISTRY,package)
            
    def _populate(self,package_type,package):
        package_type = '' if type is None else str(package_type)
        package_type = package_type.replace('\r', '\\r').replace('\n', '\\n')
        package = '' if package is None else str(package)
        package = package.replace('\r', '\\r').replace('\n', '\\n')
        self._tree.insert('',tk.END, text=package_type, values=[package])
            
    def has_selection(self):
        return self._tree.selection()
            
    def _on_tree_select(self,event):
        for item in self._tree.selection():
            self.show_remove_button(True)
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
            package_type = self._tree.item(item, "text")
            package = self._tree.item(item,'values')[0]
            self._popup_widget = PackagePopup(self,
                                package_type,
                                self._tree,
                                package,
                                state=tk.NORMAL)
            self._popup_widget.selectAll()
            self._popup_widget.place(x=x, y=y+pady, anchor=tk.W, width=width)
        
    def onadd(self):
        dialog = PackageComboDialog(self.master,self)
        dialog.do_init(tk.LEFT)
        dialog.do_modal()
        if dialog.result is not None and self._preferences.add_package(dialog.result[0],dialog.result[1]):
            self.populate()
            self.show_remove_button(self.has_selection())
            
    def onremove(self):
        for item in self._tree.selection():
            package_type = self._tree.item(item,'text')
            package = self._tree.item(item,'values')[0]
            if self._preferences.remove_package(package_type,package):
                self.populate()
                self.show_remove_button(self.has_selection())
              
            break
        
    def on_package_set(self,package_type,old_package,new_package):
        new_package = new_package.strip()
        if len(new_package) == 0:
            return False
        
        if self._preferences.change_package(package_type,old_package,new_package):
                 self.populate()
                 self.show_remove_button(self.has_selection()) 
                 return True

        return False
    
    def is_valid(self):
        return True
    
    def validate(self):
        return True
        
    def apply(self,preferences):
        changed = False
        packages = self._preferences.get_packages(Preferences.PACKAGE_TYPE_DRIVERS,[])
        if packages != preferences.get_packages(Preferences.PACKAGE_TYPE_DRIVERS,[]):
            preferences.set_packages(Preferences.PACKAGE_TYPE_DRIVERS,packages)
            changed = True
            
        packages = self._preferences.get_packages(Preferences.PACKAGE_TYPE_CHEMISTRY,[])
        if packages != preferences.get_packages(Preferences.PACKAGE_TYPE_CHEMISTRY,[]):
            preferences.set_packages(Preferences.PACKAGE_TYPE_CHEMISTRY,packages)
            changed = True
        
        if changed:
            preferences.save()
            refresh_operators()
            configuration_mgr = ConfigurationManager()
            configuration_mgr.refresh_drivers()
            
class PackagePopup(EntryCustom):

    def __init__(self, controller,package_type,parent, text, **options):
        ''' If relwidth is set, then width is ignored '''
        super(PackagePopup, self).__init__(parent,**options)
        self._controller = controller
        self._package_type = package_type
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
            valid = self._controller.on_package_set(self._package_type,self._text,new_text)
            self._text = new_text
          
        if valid:
            self.destroy()
        else:
            self.selectAll() 
            
class PackageComboDialog(Dialog):
    
    def __init__(self,parent,controller):
        super(PackageComboDialog, self).__init__(None,parent,"New Package")
        self._package_type = None
        self._package = None
        self._controller = controller
       
    def body(self, parent,options):
        ttk.Label(parent,
                  text='Type:',
                  borderwidth=0,
                  anchor=tk.E).grid(padx=7,pady=6,row=0,sticky='nse')
        self._package_type = ttk.Combobox(parent,
                                  exportselection=0,
                                  state='readonly',
                                  values=[Preferences.PACKAGE_TYPE_DRIVERS,Preferences.PACKAGE_TYPE_CHEMISTRY])
        self._package_type.current(0)
        self._package_type.grid(padx=(0,7),pady=6,row=0, column=1,sticky='nsw')
        
        ttk.Label(parent,
                  text="Package:",
                  borderwidth=0,
                  anchor=tk.E).grid(padx=7,pady=6,row=1,sticky='nse')
        self._package = EntryCustom(parent,state=tk.NORMAL)
        self._package.grid(padx=(0,7),pady=6,row=1, column=1,sticky='nsw')
        return self._package_type # initial focus
    
    def validate(self):
        package_type = self._package_type.get()
        package = self._package.get().strip()
        if len(package) == 0 or package in self._controller._preferences.get_packages(package_type,[]):
            self.initial_focus = self._package
            return False
                
        self.initial_focus = self._package_type
        return True

    def apply(self):
        self.result = (self._package_type.get(),self._package.get().strip())