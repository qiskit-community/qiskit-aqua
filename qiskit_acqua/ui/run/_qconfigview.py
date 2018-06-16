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
from qiskit_acqua.ui.run._customwidgets import EntryCustom
from qiskit_acqua.ui.run._toolbarview import ToolbarView
from qiskit_acqua.preferences import Preferences
from qiskit_acqua.ui.run._dialog import Dialog

class QconfigView(ttk.Frame):
     
    def __init__(self, parent,**options):
        super(QconfigView, self).__init__(parent, **options)
       
        self.pack(fill=tk.BOTH, expand=tk.TRUE)
        
        style = ttk.Style()
        bg = ttk.Style().lookup('TFrame', 'background')
        style = ttk.Style()
        style.configure("TNotebook", background=bg, borderwidth=0)
        self._notebook = ttk.Notebook(self,style='TNotebook')
        self._notebook.pack(fill=tk.BOTH, expand=tk.TRUE)
        
        self._mainpage = MainPage(self._notebook)
        self._proxiespage = ProxiesPage(self._notebook)
        self._notebook.add(self._mainpage, text='Main',sticky='nsew')
        self._notebook.add(self._proxiespage, text='Proxies',sticky='nsew')
        
        self.update_idletasks()
        self._notebook.configure(height=self._mainpage.winfo_reqheight())
        
class MainPage(ttk.Frame):
     
    def __init__(self, parent,**options):
        super(MainPage, self).__init__(parent, **options)
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
        self.providerEntry = None
        self._provider_name = tk.StringVar()
        self._config_path = tk.StringVar()
        
        self.pack(fill=tk.BOTH, expand=tk.TRUE)
      
        preferences = Preferences()
        self._apiToken.set(preferences.get_token('')) 
        self._url.set(preferences.get_url(Preferences.URL)) 
        self._hub.set(preferences.get_hub('')) 
        self._group.set(preferences.get_group('')) 
        self._project.set(preferences.get_project(''))
        self._provider_name.set(preferences.get_provider_name(''))
        self._config_path.set(preferences.get_qconfig_path(''))
        
        ttk.Label(self,
                  text="Token:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=0, column=0,sticky='nsew')
        self._apiTokenEntry = EntryCustom(self,
                                          textvariable=self._apiToken,
                                          width=120,
                                          state=tk.NORMAL)
        self._apiTokenEntry.grid(row=0, column=1,sticky='nsew')
        ttk.Label(self,
                  text="URL:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=1, column=0,sticky='nsew')
        self._urlEntry = EntryCustom(self,
                                     textvariable=self._url,
                                     width=60,
                                     state=tk.NORMAL)
        self._urlEntry.grid(row=1,column=1,sticky='nsw')
        ttk.Label(self,
                  text="Hub:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=2, column=0,sticky='nsew')
        self._hubEntry = EntryCustom(self,
                                     textvariable=self._hub,
                                     state=tk.NORMAL)
        self._hubEntry.grid(row=2,column=1,sticky='nsw')
        ttk.Label(self,
                  text="Group:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=3, column=0,sticky='nsew')
        self._groupEntry = EntryCustom(self,
                                       textvariable=self._group,
                                       state=tk.NORMAL)
        self._groupEntry.grid(row=3, column=1,sticky='nsw')
        ttk.Label(self,
                  text="Project:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=4, column=0,sticky='nsew')
        self._projectEntry = EntryCustom(self,
                                         textvariable=self._project,
                                         state=tk.NORMAL)
        self._projectEntry.grid(row=4, column=1,sticky='nsw')
        ttk.Label(self,
                  text="Provider:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=5, column=0,sticky='nsew')
        self._providerEntry = EntryCustom(self,
                                         textvariable=self._provider_name,
                                         state=tk.NORMAL)
        self._providerEntry.grid(row=5, column=1,sticky='nsw')
        ttk.Label(self,
                  text="Path:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=6, column=0,sticky='nsew')
        ttk.Label(self,
                  textvariable=self._config_path,
                  borderwidth=0,
                  anchor=tk.W).grid(row=6, column=1, sticky='nsw')
        
    def apply(self):
        token = self._apiToken.get().strip()
        url = self._url.get().strip()
        hub = self._hub.get().strip()
        group = self._group.get().strip()
        project = self._project.get().strip()
        provider_name = self._provider_name.get().strip()
        
        preferences = Preferences()
        preferences.set_token(token if len(token) > 0 else None)
        preferences.set_url(url if len(url) > 0 else None)
        preferences.set_hub(hub if len(hub) > 0 else None)
        preferences.set_group(group if len(group) > 0 else None)
        preferences.set_project(project if len(project) > 0 else None)
        preferences.set_provider_name(provider_name if len(provider_name) > 0 else None)
        preferences.save()
    
class ProxiesPage(ToolbarView):

    def __init__(self, parent, **options):
        super(ProxiesPage, self).__init__(parent, **options)
        self._tree = ttk.Treeview(self, selectmode=tk.BROWSE, columns=['value'])
        self._tree.heading('#0', text='Protocol')
        self._tree.heading('value',text='URL')
        self._tree.column('#0',minwidth=0,width=150,stretch=tk.NO)
        self._tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        self._tree.bind('<Button-1>', self._on_tree_edit)
        self.init_widgets(self._tree)
        
        preferences = Preferences()
        self._proxies = preferences.get_proxies()
        self._popup_widget = None
        self.pack(fill=tk.BOTH, expand=tk.TRUE)
        self.populate()
        self.show_add_button(True)
        self.show_remove_button(self.has_selection())
        self.show_defaults_button(False)
        
    def clear(self):
        if self._popup_widget is not None and self._popup_widget.winfo_exists():
            self._popup_widget.destroy()
            
        self._popup_widget = None
        for i in self._tree.get_children():
            self._tree.delete([i])
            
    def populate(self):
        self.clear()
        for protocol,url in self._proxies.items():
            url = '' if url is None else str(url)
            url = url.replace('\r', '\\r').replace('\n', '\\n')
            self._tree.insert('',tk.END, text=protocol, values=[url])
            
    def set_proxy(self,protocol,url):
        for item in self._tree.get_children():
            p = self._tree.item(item, "text")
            if p == protocol:
                self._tree.item(item, values=[url])
                break
            
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
            protocol = self._tree.item(item, "text")
            self._popup_widget = URLPopup(self,
                                protocol,
                                self._tree,
                                self._proxies[protocol],
                                state=tk.NORMAL)
            self._popup_widget.selectAll()
            self._popup_widget.place(x=x, y=y+pady, anchor=tk.W, width=width)
        
    def onadd(self):
        dialog = ProxyEntryDialog(self.master,self)
        dialog.do_init(tk.LEFT)
        dialog.do_modal()
        if dialog.result is None:
            return
        
        if dialog.result is not None:
            self._proxies[dialog.result[0]] = dialog.result[1]
            self.populate()
            self.show_remove_button(self.has_selection())
            
    def onremove(self):
        for item in self._tree.selection():
            protocol = self._tree.item(item,'text')
            if protocol in self._proxies:
                del self._proxies[protocol]
                self.populate()
                self.show_remove_button(self.has_selection())
            break
        
    def on_proxy_set(self,protocol,url):
        url = url.strip()
        if len(url) == 0:
            return False
        
        self._proxies[protocol] = url
        self.populate()
        self.show_remove_button(self.has_selection()) 
        return True
        
    def apply(self):
        preferences = Preferences()
        preferences.set_proxies(self._proxies)
        preferences.save()
        
        
class URLPopup(EntryCustom):

    def __init__(self, controller,protocol,parent, url, **options):
        ''' If relwidth is set, then width is ignored '''
        super(URLPopup, self).__init__(parent,**options)
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
            valid = self._controller.on_proxy_set(self._protocol,new_url)
        if valid:
            self.destroy()
        else:
            self.selectAll() 
        
class ProxyEntryDialog(Dialog):
    
    def __init__(self,parent,controller):
        super(ProxyEntryDialog, self).__init__(None,parent,"New Proxy")
        self._protocol = None
        self._url = None
        self._controller = controller
       
    def body(self, parent,options):
        ttk.Label(parent,
                  text="Protocol:",
                  borderwidth=0,
                  anchor=tk.E).grid(padx=7,pady=6,row=0,sticky='nse')
        self._protocol = EntryCustom(parent,state=tk.NORMAL)
        self._protocol.grid(padx=(0,7),pady=6,row=0, column=1,sticky='nsw')
        ttk.Label(parent,
                  text="URL:",
                  borderwidth=0,
                  anchor=tk.E).grid(padx=7,pady=6,row=1,sticky='nse')
        self._url = EntryCustom(parent,state=tk.NORMAL,width=50)
        self._url.grid(padx=(0,7),pady=6,row=1, column=1,sticky='nsew')
        return self._protocol # initial focus
    
    def validate(self):
        protocol = self._protocol.get().strip()
        if len(protocol) == 0 or protocol in self._controller._proxies:
            self.initial_focus = self._protocol
            return False
        
        url = self._url.get().strip()
        if len(url) == 0:
            self.initial_focus = self._url
            return False
                
        self.initial_focus = self._protocol
        return True
    
    def apply(self):
        self.result = (self._protocol.get().strip(),self._url.get().strip())
