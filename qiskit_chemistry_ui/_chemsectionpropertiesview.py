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
from qiskit_aqua_ui import SectionPropertiesView, TextPopup


class ChemSectionPropertiesView(SectionPropertiesView):

    def __init__(self, controller, parent, **options):
        super(ChemSectionPropertiesView, self).__init__(controller, parent, **options)

    def populate(self, properties):
        self.clear()
        for property_name, value_tuple in properties.items():
            value = '' if value_tuple[0] is None else str(value_tuple[0])
            value = value.replace('\r', '\\r').replace('\n', '\\n')
            if value_tuple[1]:
                self._tree.insert('', tk.END, text=property_name, values=[], tags="SUBSTITUTIONS")
            else:
                self._tree.insert('', tk.END, text=property_name, values=[value])

        self._tree.tag_configure('SUBSTITUTIONS', foreground='gray')
        self._properties = properties

    def _on_tree_edit(self, event):
        rowid = self._tree.identify_row(event.y)
        if not rowid:
            return

        column = self._tree.identify_column(event.x)
        if column == '#1':
            x, y, width, height = self._tree.bbox(rowid, column)
            pady = height // 2

            item = self._tree.identify("item", event.x, event.y)
            property_name = self._tree.item(item, "text")
            value_tuple = self._properties[property_name]
            if not value_tuple[1]:
                self._popup_widget = self._controller.create_popup(self.section_name,
                                                                   property_name,
                                                                   self._tree,
                                                                   value_tuple[0])
                if isinstance(self._popup_widget, TextPopup):
                    height = self._tree.winfo_height() - y
                    self._popup_widget.place(x=x, y=y, width=width, height=height)
                else:
                    self._popup_widget.place(x=x, y=y + pady, anchor=tk.W, width=width)
