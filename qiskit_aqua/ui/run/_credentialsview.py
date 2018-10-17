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
from tkinter import messagebox
from qiskit_aqua.ui.run._customwidgets import EntryCustom
from qiskit_aqua.ui.run._toolbarview import ToolbarView
from qiskit_aqua.preferences import Preferences
from qiskit_aqua.ui.run._dialog import Dialog
import urllib


class CredentialsView(ttk.Frame):

    def __init__(self, parent, **options):
        super(CredentialsView, self).__init__(parent, **options)

        self.pack(fill=tk.BOTH, expand=tk.TRUE)

        self._preferences = Preferences()

        ttk.Label(self,
                  text="URL:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=0, column=0, pady=5, sticky='nsew')
        urls = [
            credentials.url for credentials in self._preferences.credentials_preferences.get_all_credentials()]
        self._urlCombobox = URLCombobox(self,
                                        self,
                                        width=80,
                                        exportselection=0,
                                        state='readonly',
                                        values=urls)
        self._urlCombobox.set(self._preferences.get_url(''))
        if len(urls) > 0:
            if self._urlCombobox._text in urls:
                self._urlCombobox.current(
                    urls.index(self._urlCombobox._text))
            else:
                self._urlCombobox.current(0)

        self._urlCombobox.grid(row=0, column=1, pady=5, sticky='nsew')

        button_container = tk.Frame(self)
        button_container.grid(row=0, column=2, pady=5, sticky='nsw')
        self._add_button = ttk.Button(button_container,
                                      text='Add',
                                      state='enable',
                                      command=self.onadd)
        self._remove_button = ttk.Button(button_container,
                                         text='Remove',
                                         state='enable' if len(urls) > 0 else 'disable',
                                         command=self.onremove)
        self._add_button.pack(side=tk.LEFT)
        self._remove_button.pack(side=tk.LEFT)

        self._apiToken = tk.StringVar()
        self._apiToken.set(self._preferences.get_token(''))
        ttk.Label(self,
                  text="Token:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=1, column=0, pady=5, sticky='nsew')
        self._apiTokenEntry = EntryCustom(self,
                                          textvariable=self._apiToken,
                                          width=120,
                                          state=tk.NORMAL if len(urls) > 0 else tk.DISABLED)
        self._apiTokenEntry.grid(row=1, column=1, columnspan=2, pady=5, sticky='nsew')

        ttk.Label(self,
                  text="Proxies:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=2, column=0, pady=5, sticky='nsew')
        self._proxiespage = ProxiesPage(self, self._preferences)
        self._proxiespage.grid(row=3, column=0, columnspan=3, pady=5, sticky='nsew')
        self._proxiespage.show_add_button(True)
        self._proxiespage.show_remove_button(self._proxiespage.has_selection())
        self._proxiespage.show_defaults_button(False)
        if len(urls) == 0:
            self._proxiespage.enable(False)

        self.initial_focus = self._urlCombobox

    def onadd(self):
        dialog = URLEntryDialog(self.master, self)
        dialog.do_init(tk.LEFT)
        dialog.do_modal()
        if dialog.result is None:
            return

        credentials = self._preferences.credentials_preferences.set_credentials(
            '', dialog.result)
        self._preferences.credentials_preferences.select_credentials(
            credentials.url)
        urls = [
            credentials.url for credentials in self._preferences.credentials_preferences.get_all_credentials()]
        self._urlCombobox.config(values=urls)
        self._urlCombobox.set(self._preferences.get_url(''))
        if len(urls) > 0:
            if self._urlCombobox._text in urls:
                self._urlCombobox.current(urls.index(self._urlCombobox._text))
            else:
                self._urlCombobox.current(0)

            self._remove_button.config(state="enable")
            self._apiTokenEntry.config(state=tk.NORMAL)
            self._proxiespage.enable(True)

        self._apiToken.set(self._preferences.get_token(''))
        self._proxiespage._proxy_urls = self._preferences.get_proxy_urls({})
        self._proxiespage.populate()

    def onremove(self):
        self._preferences.credentials_preferences.remove_credentials(
            self._urlCombobox.get().strip())
        urls = [
            credentials.url for credentials in self._preferences.credentials_preferences.get_all_credentials()]
        self._urlCombobox.config(values=urls)
        self._urlCombobox.set(self._preferences.get_url(''))
        if len(urls) > 0:
            if self._urlCombobox._text in urls:
                self._urlCombobox.current(
                    urls.index(self._urlCombobox._text))
            else:
                self._urlCombobox.current(0)

        self._apiToken.set(self._preferences.get_token(''))
        self._proxiespage._proxy_urls = self._preferences.get_proxy_urls({})
        self._proxiespage.populate()

        if len(urls) == 0:
            self._remove_button.config(state="disable")
            self._apiTokenEntry.config(state=tk.DISABLED)
            self._proxiespage.enable(False)

    def on_url_set(self, url):
        credentials = self._preferences.credentials_preferences.set_credentials(
            '', url)
        self._preferences.credentials_preferences.select_credentials(
            credentials.url)
        self._apiToken.set(self._preferences.get_token(''))
        self._proxiespage._proxy_urls = self._preferences.get_proxy_urls({})
        self._proxiespage.populate()

    def is_valid(self):
        return self._proxiespage.is_valid()

    def validate(self):
        if not self._proxiespage.is_valid():
            self.initial_focus = self._proxiespage.initial_focus
            return False

        self._proxiespage.validate()
        self.initial_focus = self._urlCombobox
        return True

    def apply(self, preferences):
        pass
        # token = self._apiToken.get().strip()
        # url = self._url.get().strip()

        # preferences.set_token(token if len(token) > 0 else None)
        # preferences.set_url(url if len(url) > 0 else None)
        # self._proxiespage.apply(preferences)

    @staticmethod
    def _is_valid_url(url):
        if url is None or not isinstance(url, str):
            return False

        url = url.strip()
        if len(url) == 0:
            return False

        min_attributes = ('scheme', 'netloc')
        valid = True
        try:
            token = urllib.parse.urlparse(url)
            if not all([getattr(token, attr) for attr in min_attributes]):
                valid = False
        except:
            valid = False

        return valid

    @staticmethod
    def _validate_url(url):
        valid = CredentialsView._is_valid_url(url)
        if not valid:
            messagebox.showerror("Error", 'Invalid url')

        return valid


class URLCombobox(ttk.Combobox):

    def __init__(self, controller, parent, **options):
        ''' If relwidth is set, then width is ignored '''
        super(URLCombobox, self).__init__(parent, **options)
        self._controller = controller
        self.bind("<<ComboboxSelected>>", self._on_select)
        self._text = None

    def _on_select(self, *ignore):
        new_text = self.get()
        if len(new_text) > 0 and self._text != new_text:
            self._text = new_text
            self._controller.on_url_set(new_text)


class URLEntryDialog(Dialog):

    def __init__(self, parent, controller):
        super(URLEntryDialog, self).__init__(None, parent, "New URL")
        self._url = None
        self._controller = controller

    def body(self, parent, options):
        ttk.Label(parent,
                  text="URL:",
                  borderwidth=0,
                  anchor=tk.E).grid(padx=7, pady=6, row=0, sticky='nse')
        self._url = EntryCustom(parent, state=tk.NORMAL, width=50)
        self._url.insert(0, Preferences.URL)
        self._url.grid(padx=(0, 7), pady=6, row=0, column=1, sticky='nsew')
        return self._url  # initial focus

    def validate(self):
        url = self._url.get().strip()
        if not CredentialsView._validate_url(url):
            self.initial_focus = self._url
            return False

        self.initial_focus = self._url
        return True

    def apply(self):
        self.result = self._url.get().strip()


class ProxiesPage(ToolbarView):

    def __init__(self, parent, preferences, **options):
        super(ProxiesPage, self).__init__(parent, **options)
        size = font.nametofont('TkHeadingFont').actual('size')
        ttk.Style().configure("ProxiesPage.Treeview.Heading", font=(None, size, 'bold'))
        self._tree = ttk.Treeview(
            self, style='ProxiesPage.Treeview', selectmode=tk.BROWSE, height=3, columns=['value'])
        self._tree.heading('#0', text='Protocol')
        self._tree.heading('value', text='URL')
        self._tree.column('#0', minwidth=0, width=150, stretch=tk.NO)
        self._tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        self._tree.bind('<Button-1>', self._on_tree_edit)
        self.init_widgets(self._tree)

        self._proxy_urls = preferences.get_proxy_urls({})
        self._popup_widget = None
        self.populate()
        self.initial_focus = self._tree

    def enable(self, enable=True):
        if enable and "disabled" in self._tree.state():
            self._tree.state(("!disabled",))
            self.show_add_button(True)
            self.show_remove_button(self.has_selection())
            return

        if not enable and "disabled" not in self._tree.state():
            self._tree.state(("disabled",))
            self.show_add_button(False)
            self.show_remove_button(False)

    def clear(self):
        if self._popup_widget is not None and self._popup_widget.winfo_exists():
            self._popup_widget.destroy()

        self._popup_widget = None
        for i in self._tree.get_children():
            self._tree.delete([i])

    def populate(self):
        self.clear()
        for protocol, url in self._proxy_urls.items():
            url = '' if url is None else str(url)
            url = url.replace('\r', '\\r').replace('\n', '\\n')
            self._tree.insert('', tk.END, text=protocol, values=[url])

    def set_proxy(self, protocol, url):
        for item in self._tree.get_children():
            p = self._tree.item(item, "text")
            if p == protocol:
                self._tree.item(item, values=[url])
                break

    def has_selection(self):
        return self._tree.selection()

    def _on_tree_select(self, event):
        for item in self._tree.selection():
            self.show_remove_button(True)
            return

    def _on_tree_edit(self, event):
        if 'disabled' in self._tree.state():
            return

        rowid = self._tree.identify_row(event.y)
        if not rowid:
            return

        column = self._tree.identify_column(event.x)
        if column == '#1':
            x, y, width, height = self._tree.bbox(rowid, column)
            pady = height // 2

            item = self._tree.identify("item", event.x, event.y)
            protocol = self._tree.item(item, "text")
            self._popup_widget = URLPopup(self,
                                          protocol,
                                          self._tree,
                                          self._proxy_urls[protocol],
                                          state=tk.NORMAL)
            self._popup_widget.selectAll()
            self._popup_widget.place(x=x, y=y + pady, anchor=tk.W, width=width)

    def onadd(self):
        dialog = ProxyEntryDialog(self.master, self)
        dialog.do_init(tk.LEFT)
        dialog.do_modal()
        if dialog.result is None:
            return

        if dialog.result is not None:
            self._proxy_urls[dialog.result[0]] = dialog.result[1]
            self.populate()
            self.show_remove_button(self.has_selection())

    def onremove(self):
        for item in self._tree.selection():
            protocol = self._tree.item(item, 'text')
            if protocol in self._proxy_urls:
                del self._proxy_urls[protocol]
                self.populate()
                self.show_remove_button(self.has_selection())
            break

    def on_proxy_set(self, protocol, url):
        protocol = protocol.strip()
        if len(protocol) == 0:
            return False

        url = url.strip()
        if not CredentialsView._validate_url(url):
            return False

        self._proxy_urls[protocol] = url
        self.populate()
        self.show_remove_button(self.has_selection())
        return True

    def is_valid(self):
        return True

    def validate(self):
        return True

    def apply(self, preferences):
        preferences.set_proxy_urls(
            self._proxy_urls if len(self._proxy_urls) > 0 else None)


class URLPopup(EntryCustom):

    def __init__(self, controller, protocol, parent, url, **options):
        ''' If relwidth is set, then width is ignored '''
        super(URLPopup, self).__init__(parent, **options)
        self._controller = controller
        self._protocol = protocol
        self._url = url
        self.insert(0, self._url)
        self.focus_force()
        self.bind("<Unmap>", self._update_value)
        self.bind("<FocusOut>", self._update_value)

    def selectAll(self):
        self.focus_force()
        self.selection_range(0, tk.END)

    def _update_value(self, *ignore):
        new_url = self.get().strip()
        valid = True
        if self._url != new_url:
            self._url = new_url
            valid = self._controller.on_proxy_set(self._protocol, new_url)
        if valid:
            self.destroy()
        else:
            self.selectAll()


class ProxyEntryDialog(Dialog):

    def __init__(self, parent, controller):
        super(ProxyEntryDialog, self).__init__(None, parent, "New Proxy")
        self._protocol = None
        self._url = None
        self._controller = controller

    def body(self, parent, options):
        ttk.Label(parent,
                  text="Protocol:",
                  borderwidth=0,
                  anchor=tk.E).grid(padx=7, pady=6, row=0, sticky='nse')
        self._protocol = EntryCustom(parent, state=tk.NORMAL)
        self._protocol.grid(padx=(0, 7), pady=6, row=0, column=1, sticky='nsw')
        ttk.Label(parent,
                  text="URL:",
                  borderwidth=0,
                  anchor=tk.E).grid(padx=7, pady=6, row=1, sticky='nse')
        self._url = EntryCustom(parent, state=tk.NORMAL, width=50)
        self._url.grid(padx=(0, 7), pady=6, row=1, column=1, sticky='nsew')
        return self._protocol  # initial focus

    def validate(self):
        protocol = self._protocol.get().strip()
        if len(protocol) == 0 or protocol in self._controller._proxy_urls:
            self.initial_focus = self._protocol
            return False

        url = self._url.get().strip()
        if not CredentialsView._validate_url(url):
            self.initial_focus = self._url
            return False

        self.initial_focus = self._protocol
        return True

    def apply(self):
        self.result = (self._protocol.get().strip(), self._url.get().strip())
