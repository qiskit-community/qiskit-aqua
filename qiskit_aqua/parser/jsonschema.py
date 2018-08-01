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

import json
import os
import jsonschema

class JSONSchema(object):
    """JSON schema Utilities class."""
    
    def __init__(self, jsonfile, resolve_references = False):
        """Create JSONSchema object."""

        with open(jsonfile) as json_file:
            self._schema = json.load(json_file)
            if resolve_references:
                validator = jsonschema.Draft4Validator(self._schema)
                self._schema = JSONSchema._resolve_schema_references(validator.schema,validator.resolver)
            
    @property
    def schema(self):
        """Return json schema"""
        return self._schema
          
    @staticmethod
    def load_algorithms_main_schema(resolve_references = False):
        """Returns a JSONSchema instance with algorithms input schema"""
        return JSONSchema(os.path.join(os.path.dirname(__file__), 'input_schema.json'),resolve_references)
    
    @staticmethod
    def _resolve_schema_references(schema, resolver):
        """
        Resolves json references and merges them into the schema
        Params:
            schema (dict): schema
            resolver (ob): Validator Resolver
            
        Returns schema merged with resolved references
        """
        if isinstance(schema, dict):
            for key, value in schema.items():
                if key == '$ref':
                    ref_schema = resolver.resolve(value)
                    if ref_schema:
                        return ref_schema[1]

                resolved_ref = JSONSchema._resolve_schema_references(value, resolver)
                if resolved_ref:
                    schema[key] = resolved_ref
                    
        elif isinstance(schema, list):
            for (idx, value) in enumerate(schema):
                resolved_ref = JSONSchema._resolve_schema_references(value, resolver)
                if resolved_ref:
                    schema[idx] = resolved_ref
                    
        return schema
    