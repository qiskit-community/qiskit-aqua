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
from qiskit_acqua_chemistry.ui._dialog import Dialog
from collections import OrderedDict
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
        self._label_text = None
        self._label = None
        self._apiTokenEntry = None
        self._apiToken = tk.StringVar()
        self._urlEntry = None
        self._url = tk.StringVar()
        self._hubEntry = None
        self._hub = tk.StringVar()
        self._groupEntry = None
        self._group = tk.StringVar()
        self._projectEntry = None
        self._project = tk.StringVar()
        self._config_path = tk.StringVar()
        self._levelCombo = None
        self._checkButton = None
        self._populateDefaults = tk.IntVar()
       
    def body(self,parent,options):
        preferences = Preferences()
        logging_config = preferences.get_logging_config()
        if logging_config is not None:
            set_logger_config(logging_config)
        
        self._apiToken.set(preferences.get_token('')) 
        self._url.set(preferences.get_url(Preferences.URL)) 
        self._hub.set(preferences.get_hub('')) 
        self._group.set(preferences.get_group('')) 
        self._project.set(preferences.get_project(''))
        self._config_path.set(preferences.get_qconfig_path(''))
        uipreferences = UIPreferences()
        populate = uipreferences.get_populate_defaults(True)
        self._populateDefaults.set(1 if populate else 0)
        
        qiskitGroup = ttk.LabelFrame(parent,
                                     text='QISKit Configuration',
                                     padding=(6,6,6,6),
                                     borderwidth=4,
                                     relief=tk.GROOVE)
        qiskitGroup.grid(padx=(7,7),pady=6,row=0, column=0,sticky='nsew')
        qiskitGroup.columnconfigure(0,weight=1)
        qiskitGroup.columnconfigure(1,pad=7)
        ttk.Label(qiskitGroup,
                  text="Token:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=0, column=0,sticky='nsew')
        self._apiTokenEntry = EntryCustom(qiskitGroup,
                                          textvariable=self._apiToken,
                                          width=120,
                                          state=tk.NORMAL)
        self._apiTokenEntry.grid(row=0, column=1,sticky='nsew')
        ttk.Label(qiskitGroup,
                  text="URL:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=1, column=0,sticky='nsew')
        self._urlEntry = EntryCustom(qiskitGroup,
                                     textvariable=self._url,
                                     width=60,
                                     state=tk.NORMAL)
        self._urlEntry.grid(row=1,column=1,sticky='nsw')
        ttk.Label(qiskitGroup,
                  text="Hub:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=2, column=0,sticky='nsew')
        self._hubEntry = EntryCustom(qiskitGroup,
                                     textvariable=self._hub,
                                     state=tk.NORMAL)
        self._hubEntry.grid(row=2,column=1,sticky='nsw')
        ttk.Label(qiskitGroup,
                  text="Group:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=3, column=0,sticky='nsew')
        self._groupEntry = EntryCustom(qiskitGroup,
                                       textvariable=self._group,
                                       state=tk.NORMAL)
        self._groupEntry.grid(row=3, column=1,sticky='nsw')
        ttk.Label(qiskitGroup,
                  text="Project:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=4, column=0,sticky='nsew')
        self._projectEntry = EntryCustom(qiskitGroup,
                                         textvariable=self._project,
                                         state=tk.NORMAL)
        self._projectEntry.grid(row=4, column=1,sticky='nsw')
        ttk.Label(qiskitGroup,
                  text="Path:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=5, column=0,sticky='nsew')
        ttk.Label(qiskitGroup,
                  textvariable=self._config_path,
                  borderwidth=0,
                  anchor=tk.W).grid(row=5, column=1, sticky='nsw')
        
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
        
        loggingGroup = ttk.LabelFrame(parent,
                                     text='Logging Configuration',
                                     padding=(6,6,6,6),
                                     borderwidth=4,
                                     relief=tk.GROOVE)
        loggingGroup.grid(padx=(7,7),pady=6,row=2, column=0,sticky='nsw')
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
        
        self._label_text = tk.StringVar()
        self._label = ttk.Label(parent,foreground='red',
                               textvariable=self._label_text,
                               borderwidth=0)
        self._label.grid(padx=(7,7),
                        pady=6,
                        row=2,
                        column=0)
        self._label.grid_remove()
        
        self.entry = self._apiTokenEntry
        return self.entry # initial focus
    
    def validate(self):
        self._label.grid_remove()
        return True

    def apply(self):
        level_name = self._levelCombo.get()
        levels = [key for key, value in PreferencesDialog._LOG_LEVELS.items() if value == level_name]
        loglevel = levels[0]
        logging_config = build_logging_config(['qiskit_acqua_chemistry','qiskit_acqua'],loglevel)
        
        token = self._apiToken.get().strip()
        url = self._url.get().strip()
        hub = self._hub.get().strip()
        group = self._group.get().strip()
        project = self._project.get().strip()
        
        preferences = Preferences()
        preferences.set_token(token if len(token) > 0 else None)
        preferences.set_url(url if len(url) > 0 else None)
        preferences.set_hub(hub if len(hub) > 0 else None)
        preferences.set_group(group if len(group) > 0 else None)
        preferences.set_project(project if len(project) > 0 else None)
        preferences.set_logging_config(logging_config)
        preferences.save()
        set_logger_config(logging_config)
        
        uipreferences = UIPreferences()
        populate = self._populateDefaults.get()
        uipreferences.set_populate_defaults(False if populate == 0 else True)
        uipreferences.save()
        
        self._controller.get_available_backends()
