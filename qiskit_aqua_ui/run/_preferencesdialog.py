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
from ._dialog import Dialog
from collections import OrderedDict
from ._credentialsview import CredentialsView
from qiskit_aqua_ui._uipreferences import UIPreferences
import logging


class PreferencesDialog(Dialog):

    _LOG_LEVELS = OrderedDict(
        [(logging.CRITICAL, logging.getLevelName(logging.CRITICAL)),
         (logging.ERROR, logging.getLevelName(logging.ERROR)),
         (logging.WARNING, logging.getLevelName(logging.WARNING)),
         (logging.INFO, logging.getLevelName(logging.INFO)),
         (logging.DEBUG, logging.getLevelName(logging.DEBUG)),
         (logging.NOTSET, logging.getLevelName(logging.NOTSET))]
    )

    def __init__(self, controller, parent):
        super(PreferencesDialog, self).__init__(
            controller, parent, 'Preferences')
        self._credentialsview = None
        self._levelCombo = None
        self._checkButton = None
        self._populateDefaults = tk.IntVar()

    def body(self, parent, options):
        from qiskit_aqua._logging import (get_logging_level,
                                          set_logging_config)
        from qiskit_aqua_cmd import Preferences
        preferences = Preferences()
        logging_config = preferences.get_logging_config()
        if logging_config is not None:
            set_logging_config(logging_config)

        uipreferences = UIPreferences()
        populate = uipreferences.get_populate_defaults(True)
        self._populateDefaults.set(1 if populate else 0)

        credentialsGroup = ttk.LabelFrame(parent,
                                          text='IBMQ Credentials',
                                          padding=(6, 6, 6, 6),
                                          borderwidth=4,
                                          relief=tk.GROOVE)
        credentialsGroup.grid(padx=(7, 7), pady=6, row=0,
                              column=0, sticky='nsew')
        self._credentialsview = CredentialsView(credentialsGroup)

        defaultsGroup = ttk.LabelFrame(parent,
                                       text='Defaults',
                                       padding=(6, 6, 6, 6),
                                       borderwidth=4,
                                       relief=tk.GROOVE)
        defaultsGroup.grid(padx=(7, 7), pady=6, row=1, column=0, sticky='nsw')
        defaultsGroup.columnconfigure(1, pad=7)

        self._checkButton = ttk.Checkbutton(defaultsGroup,
                                            text="Populate on file new/open",
                                            variable=self._populateDefaults)
        self._checkButton.grid(row=0, column=1, sticky='nsw')

        loggingGroup = ttk.LabelFrame(parent,
                                      text='Logging Configuration',
                                      padding=(6, 6, 6, 6),
                                      borderwidth=4,
                                      relief=tk.GROOVE)
        loggingGroup.grid(padx=(7, 7), pady=6, row=2, column=0, sticky='nsw')
        loggingGroup.columnconfigure(1, pad=7)

        loglevel = get_logging_level()

        ttk.Label(loggingGroup,
                  text="Level:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=0, column=0, sticky='nsew')
        self._levelCombo = ttk.Combobox(loggingGroup,
                                        exportselection=0,
                                        state='readonly',
                                        values=list(PreferencesDialog._LOG_LEVELS.values()))
        index = list(PreferencesDialog._LOG_LEVELS.keys()).index(loglevel)
        self._levelCombo.current(index)
        self._levelCombo.grid(row=0, column=1, sticky='nsw')

        self.entry = self._credentialsview.initial_focus
        return self.entry  # initial focus

    def validate(self):
        if not self._credentialsview.validate():
            self.initial_focus = self._credentialsview.initial_focus
            return False

        self.initial_focus = self._credentialsview.initial_focus
        return True

    def apply(self):
        from qiskit_aqua_cmd import Preferences
        from qiskit_aqua import disable_ibmq_account
        from qiskit_aqua._logging import (build_logging_config,
                                          set_logging_config)
        try:
            level_name = self._levelCombo.get()
            levels = [key for key, value in PreferencesDialog._LOG_LEVELS.items() if value == level_name]
            loglevel = levels[0]

            preferences = Preferences()
            disable_ibmq_account(preferences.get_url(), preferences.get_token(), preferences.get_proxies({}))
            self._credentialsview.apply(preferences)
            preferences.save()

            logging_config = build_logging_config(loglevel)

            preferences = Preferences()
            preferences.set_logging_config(logging_config)
            preferences.save()

            set_logging_config(logging_config)

            uipreferences = UIPreferences()
            populate = self._populateDefaults.get()
            uipreferences.set_populate_defaults(False if populate == 0 else True)
            uipreferences.save()

            self._controller.model.get_available_providers()
        except Exception as e:
            self.controller.outputview.write_line(str(e))
