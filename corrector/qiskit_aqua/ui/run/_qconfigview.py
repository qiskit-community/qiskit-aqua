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

class QconfigView(ttk.Frame):
     
    def __init__(self, parent,**options):
        super(QconfigView, self).__init__(parent, **options)
       
        self.pack(fill=tk.BOTH, expand=tk.TRUE)
        
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(side=tk.TOP,fill=tk.BOTH, expand=tk.TRUE)
        
        preferences = Preferences()
        self._mainpage = MainPage(self._notebook,preferences)
        self._proxiespage = ProxiesPage(self._notebook,preferences)
        self._notebook.add(self._mainpage, text='Main')
        self._notebook.add(self._proxiespage, text='Proxies')
        self._notebook.bind('<<NotebookTabChanged>>', self._tab_changed)
        
        frame = ttk.Frame(self)
        frame.pack(side=tk.BOTTOM,fill=tk.X, expand=tk.TRUE)
        
        ttk.Label(frame,
                  text="Path:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=0, column=0,padx=6,sticky='nsew')
        ttk.Label(frame,
                  text=preferences.get_qconfig_path(''),
                  borderwidth=0,
                  anchor=tk.W).grid(row=0, column=1, sticky='nsw')
        
        self.initial_focus = self._mainpage.initial_focus
        self.update_idletasks()
        self._notebook.configure(height=self._mainpage.winfo_reqheight())
        
    def _tab_changed(self, *ignore):
        if self._notebook.index(self._notebook.select()) == 0:
            if not self._mainpage.validate():
                self.initial_focus = self._mainpage.initial_focus
                self.initial_focus.focus_set()
                return
            
            if not self._proxiespage.is_valid():
                self._notebook.select(1)
    
        if self._notebook.index(self._notebook.select()) == 1:
            if not self._proxiespage.validate():
                self.initial_focus = self._proxiespage.initial_focus
                self.initial_focus.focus_set()
                return
            
            if not self._mainpage.is_valid():
                self._notebook.select(0)
            
    def validate(self):
        if not self._mainpage.is_valid():
            if self._notebook.index(self._notebook.select()) != 0:
                self._notebook.select(0)
                return False
            
            self._mainpage.validate()
            self.initial_focus = self._mainpage.initial_focus
            return False
          
        if not self._proxiespage.is_valid():
            if self._notebook.index(self._notebook.select()) != 1:
                self._notebook.select(1)
                return False
            
            self._proxiespage.validate()
            self.initial_focus = self._mainpage.initial_focus
            return False
         
        self.initial_focus = self._mainpage.initial_focus
        return True
        
    def apply(self,preferences):
        self._mainpage.apply(preferences)
        self._proxiespage.apply(preferences)
        
        
    @staticmethod
    def _is_valid_url(url):
        if url is None or not isinstance(url,str):
            return False
        
        url = url.strip()
        if len(url) == 0:
            return False
        
        min_attributes = ('scheme','netloc')
        valid = True
        try:
            token = urllib.parse.urlparse(url)
            if not all([getattr(token,attr) for attr in min_attributes]):
                valid = False
        except:
            valid = False
       
        return valid
    
    @staticmethod
    def _validate_url(url):
        valid = QconfigView._is_valid_url(url)
        if not valid:
            messagebox.showerror("Error",'Invalid url')
      
        return valid
        
class MainPage(ttk.Frame):
     
    def __init__(self, parent, preferences,**options):
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
        self._verifyEntry = None
        
        self.pack(fill=tk.BOTH, expand=tk.TRUE)
      
        self._apiToken.set(preferences.get_token('')) 
        self._url.set(preferences.get_url(Preferences.URL)) 
        self._hub.set(preferences.get_hub('')) 
        self._group.set(preferences.get_group('')) 
        self._project.set(preferences.get_project(''))
        self._verify = preferences.get_verify(Preferences.VERIFY)
        
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
                  text="Verify:",
                  borderwidth=0,
                  anchor=tk.E).grid(row=5, column=0,sticky='nsew')
        values = ['True','False']
        self._verifyEntry = ttk.Combobox(self,
                                  exportselection=0,
                                  state='readonly',
                                  values=values,
                                  width=6)
        self._verifyEntry.current(values.index(str(self._verify)))
        self._verifyEntry.grid(row=5, column=1,sticky='nsw')
        
        self.initial_focus = self._apiTokenEntry
        
    def is_valid(self):
        return QconfigView._is_valid_url(self._url.get().strip())
        
    def validate(self):
        if not QconfigView._validate_url(self._url.get().strip()):
            self.initial_focus = self._urlEntry
            return False
    
        self.initial_focus = self._apiTokenEntry
        return True
        
    def apply(self,preferences):
        token = self._apiToken.get().strip()
        url = self._url.get().strip()
        hub = self._hub.get().strip()
        group = self._group.get().strip()
        project = self._project.get().strip()
        verify = self._verifyEntry.get().lower() == 'true'
    
        preferences.set_token(token if len(token) > 0 else None)
        preferences.set_url(url if len(url) > 0 else None)
        preferences.set_hub(hub if len(hub) > 0 else None)
        preferences.set_group(group if len(group) > 0 else None)
        preferences.set_project(project if len(project) > 0 else None)
        preferences.set_verify(verify)
    
class ProxiesPage(ToolbarView):

    def __init__(self, parent, preferences, **options):
        super(ProxiesPage, self).__init__(parent, **options)
        size = font.nametofont('TkHeadingFont').actual('size')
        ttk.Style().configure("ProxiesPage.Treeview.Heading", font=(None,size,'bold'))
        self._tree = ttk.Treeview(self, style='ProxiesPage.Treeview', selectmode=tk.BROWSE, columns=['value'])
        self._tree.heading('#0', text='Protocol')
        self._tree.heading('value',text='URL')
        self._tree.column('#0',minwidth=0,width=150,stretch=tk.NO)
        self._tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        self._tree.bind('<Button-1>', self._on_tree_edit)
        self.init_widgets(self._tree)
        
        self._proxy_urls = preferences.get_proxy_urls({})
        self._popup_widget = None
        self.pack(fill=tk.BOTH, expand=tk.TRUE)
        self.populate()
        self.show_add_button(True)
        self.show_remove_button(self.has_selection())
        self.show_defaults_button(False)
        self.initial_focus = self._tree
        
    def clear(self):
        if self._popup_widget is not None and self._popup_widget.winfo_exists():
            self._popup_widget.destroy()
            
        self._popup_widget = None
        for i in self._tree.get_children():
            self._tree.delete([i])
            
    def populate(self):
        self.clear()
        for protocol,url in self._proxy_urls.items():
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
                                self._proxy_urls[protocol],
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
            self._proxy_urls[dialog.result[0]] = dialog.result[1]
            self.populate()
            self.show_remove_button(self.has_selection())
            
    def onremove(self):
        for item in self._tree.selection():
            protocol = self._tree.item(item,'text')
            if protocol in self._proxy_urls:
                del self._proxy_urls[protocol]
                self.populate()
                self.show_remove_button(self.has_selection())
            break
        
    def on_proxy_set(self,protocol,url):
        protocol = protocol.strip()
        if len(protocol) == 0:
            return False
        
        url = url.strip()
        if not QconfigView._validate_url(url):
            return False
        
        self._proxy_urls[protocol] = url
        self.populate()
        self.show_remove_button(self.has_selection()) 
        return True
    
    def is_valid(self):
        return True
    
    def validate(self):
        return True
        
    def apply(self,preferences):
        preferences.set_proxy_urls(self._proxy_urls if len(self._proxy_urls) > 0 else None)
        
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
        if len(protocol) == 0 or protocol in self._controller._proxy_urls:
            self.initial_focus = self._protocol
            return False
        
        url = self._url.get().strip()
        if not QconfigView._validate_url(url):
            self.initial_focus = self._url
            return False
                
        self.initial_focus = self._protocol
        return True
    
    def apply(self):
        self.result = (self._protocol.get().strip(),self._url.get().strip())
