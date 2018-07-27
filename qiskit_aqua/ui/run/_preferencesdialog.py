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
from qiskit_aqua.ui.run._dialog import Dialog
from collections import OrderedDict
from qiskit_aqua.ui.run._qconfigview import QconfigView
from qiskit_aqua.ui.run._toolbarview import ToolbarView
from qiskit_aqua.ui.run._customwidgets import EntryCustom
from qiskit_aqua.preferences import Preferences
from qiskit_aqua.ui._uipreferences import UIPreferences
from qiskit_aqua import refresh_pluggables
from qiskit_aqua._logging import get_logger_levels_for_names,build_logging_config,set_logger_config
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
                                     text='Qiskit Configuration',
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
        
        levels = get_logger_levels_for_names(['qiskit_aqua'])
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
            logging_config = build_logging_config(['qiskit_aqua'],loglevel)
        
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
        self._tree = ttk.Treeview(self, style='PackagesPage.Treeview', selectmode=tk.BROWSE, height=3)
        self._tree.heading('#0', text='Name')
        self._tree.column('#0',minwidth=0,width=500,stretch=tk.NO)
        self._tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        self.init_widgets(self._tree)
        
        self._packages = preferences.get_packages([])
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
        for package in self._packages:
            package = '' if package is None else str(package)
            package = package.replace('\r', '\\r').replace('\n', '\\n')
            self._tree.insert('',tk.END, text=package)
            
    def has_selection(self):
        return self._tree.selection()
            
    def _on_tree_select(self,event):
        for item in self._tree.selection():
            self.show_remove_button(True)
            return
        
    def onadd(self):
        dialog = PackageEntryDialog(self.master,self)
        dialog.do_init(tk.LEFT)
        dialog.do_modal()
        if dialog.result is None:
            return
        
        if dialog.result is not None:
            self._packages.append(dialog.result)
            self.populate()
            self.show_remove_button(self.has_selection())
            
    def onremove(self):
        for item in self._tree.selection():
            package = self._tree.item(item,'text')
            if package in self._packages:
                self._packages.remove(package)
                self.populate()
                self.show_remove_button(self.has_selection())
            break
    
    def is_valid(self):
        return True
    
    def validate(self):
        return True
        
    def apply(self,preferences):
        if self._packages != preferences.get_packages([]):
            preferences.set_packages(self._packages if len(self._packages) > 0 else None)
            preferences.save()
            refresh_pluggables()
        
class PackageEntryDialog(Dialog):
    
    def __init__(self,parent,controller):
        super(PackageEntryDialog, self).__init__(None,parent,"New Package")
        self._package = None
        self._controller = controller
       
    def body(self, parent,options):
        ttk.Label(parent,
                  text="Package:",
                  borderwidth=0,
                  anchor=tk.E).grid(padx=7,pady=6,row=0,sticky='nse')
        self._package = EntryCustom(parent,state=tk.NORMAL)
        self._package.grid(padx=(0,7),pady=6,row=0, column=1,sticky='nsw')
        return self._package # initial focus
    
    def validate(self):
        package = self._package.get().strip()
        if len(package) == 0 or package in self._controller._packages:
            self.initial_focus = self._package
            return False
                
        self.initial_focus = self._package
        return True
    
    def apply(self):
        self.result = self._package.get().strip()

