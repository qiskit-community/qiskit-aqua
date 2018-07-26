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

import os
import json

class UIPreferences(object):
   
    _FILENAME = '.qiskit_aqua_ui'
    _VERSION = '1.0'
    
    def __init__(self):
        """Create UIPreferences object."""
        self._preferences = { 
                'version' : UIPreferences._VERSION
        }
        home = os.path.expanduser("~")
        self._filepath = os.path.join(home,UIPreferences._FILENAME)
        try:
            with open(self._filepath) as json_pref:
                self._preferences = json.load(json_pref)
        except:
            pass
            
    def save(self):
        with open(self._filepath, 'w') as fp:
            json.dump(self._preferences, fp, sort_keys=True, indent=4) 
        
    def get_version(self):
        if 'version' in self._preferences:
            return self._preferences['version']
        
        return None
            
    def get_browser_geometry(self,default_value=None):
        if 'browser_geometry' in self._preferences:
            return self._preferences['browser_geometry']
        
        return default_value
    
    def set_browser_geometry(self,geometry):
        self._preferences['browser_geometry'] = geometry
        
    def get_run_geometry(self,default_value=None):
        if 'run_geometry' in self._preferences:
            return self._preferences['run_geometry']
        
        return default_value
    
    def set_run_geometry(self,geometry):
        self._preferences['run_geometry'] = geometry
        
    def get_openfile_initialdir(self):
        if 'openfile_initialdir' in self._preferences:
            if not os.path.isdir(self._preferences['openfile_initialdir']):
                self._preferences['openfile_initialdir'] = os.getcwd()
                
            return self._preferences['openfile_initialdir']
    
        return os.getcwd()
    
    def set_openfile_initialdir(self,initialdir):
        self._preferences['openfile_initialdir'] = initialdir
        
    def get_savefile_initialdir(self):
        if 'savefile_initialdir' in self._preferences:
            if not os.path.isdir(self._preferences['savefile_initialdir']):
                self._preferences['savefile_initialdir'] = os.getcwd()
                
            return self._preferences['savefile_initialdir']
        
        return os.getcwd()
    
    def set_savefile_initialdir(self,initialdir):
        self._preferences['savefile_initialdir'] = initialdir
        
    def get_populate_defaults(self,default_value=None):
        if 'populate_defaults' in self._preferences:
            return self._preferences['populate_defaults']
        
        return default_value
    
    def set_populate_defaults(self,populate_defaults):
        self._preferences['populate_defaults'] = populate_defaults
        
    def get_recent_files(self):
        files = []
        if 'recent_files' in self._preferences:
            for file in self._preferences['recent_files']:
                if os.path.isfile(file):
                    files.append(file)
                    
            self._preferences['recent_files'] = files
        
        return files
    
    def add_recent_file(self,file):
        recent_files = self.get_recent_files()
        if file not in recent_files:
            recent_files.append(file)
            if len(recent_files) > 6:
                recent_files = recent_files[1:]
        self._preferences['recent_files'] = recent_files
        
    def clear_recent_files(self):
        if 'recent_files' in self._preferences:
            del self._preferences['recent_files']
    
