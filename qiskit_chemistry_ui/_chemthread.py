# -*- coding: utf-8 -*-

# Copyright 2019 IBM.
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

import psutil
import os
import subprocess
import threading
import tempfile
import sys
import logging
from qiskit_aqua_ui import GUIProvider
import traceback

logger = logging.getLogger(__name__)


def exception_to_string(excp):
    stack = traceback.extract_stack()[:-3] + traceback.extract_tb(excp.__traceback__)
    pretty = traceback.format_list(stack)
    return ''.join(pretty) + '\n  {} {}'.format(excp.__class__, excp)


class ChemistryThread(threading.Thread):

    def __init__(self, model, output, queue, filename):
        super(ChemistryThread, self).__init__(name='Chemistry run thread')
        self.model = model
        self._output = output
        self._thread_queue = queue
        self._json_algo_file = filename
        self._popen = None

    def stop(self):
        self._output = None
        self._thread_queue = None
        if self._popen is not None:
            p = self._popen
            self._kill(p.pid)
            p.stdout.close()

    def _kill(self, proc_pid):
        try:
            process = psutil.Process(proc_pid)
            for proc in process.children(recursive=True):
                proc.kill()
            process.kill()
        except Exception as e:
            if self._output is not None:
                self._output.write_line(
                    'Process kill has failed: {}'.format(str(e)))

    def run(self):
        input_file = None
        output_file = None
        temp_input = False
        try:
            qiskit_chemistry_directory = os.path.dirname(os.path.realpath(__file__))
            qiskit_chemistry_directory = os.path.abspath(
                os.path.join(qiskit_chemistry_directory, '../qiskit_chemistry_cmd'))
            input_file = self.model.get_filename()
            if input_file is None or self.model.is_modified():
                fd, input_file = tempfile.mkstemp(suffix='.in')
                os.close(fd)
                temp_input = True
                self.model.save_to_file(input_file)

            startupinfo = None
            process_name = psutil.Process().exe()
            if process_name is None or len(process_name) == 0:
                process_name = 'python'
            else:
                if sys.platform == 'win32' and process_name.endswith('pythonw.exe'):
                    path = os.path.dirname(process_name)
                    files = [f for f in os.listdir(path) if f != 'pythonw.exe' and f.startswith(
                        'python') and f.endswith('.exe')]
                    # sort reverse to have the python versions first: python3.exe before python2.exe
                    files = sorted(files, key=str.lower, reverse=True)
                    new_process = None
                    for file in files:
                        p = os.path.join(path, file)
                        if os.path.isfile(p):
                            # python.exe takes precedence
                            if file.lower() == 'python.exe':
                                new_process = p
                                break

                            # use first found
                            if new_process is None:
                                new_process = p

                    if new_process is not None:
                        startupinfo = subprocess.STARTUPINFO()
                        startupinfo.dwFlags = subprocess.STARTF_USESHOWWINDOW
                        startupinfo.wShowWindow = subprocess.SW_HIDE
                        process_name = new_process

            input_array = [process_name, qiskit_chemistry_directory, input_file]
            if self._json_algo_file:
                input_array.extend(['-jo', self._json_algo_file])
            else:
                fd, output_file = tempfile.mkstemp(suffix='.out')
                os.close(fd)
                input_array.extend(['-o', output_file])

            if self._output is not None and logger.getEffectiveLevel() == logging.DEBUG:
                self._output.write('Process: {}\n'.format(process_name))

            self._popen = subprocess.Popen(input_array,
                                           stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT,
                                           universal_newlines=True,
                                           startupinfo=startupinfo)
            if self._thread_queue is not None:
                self._thread_queue.put(GUIProvider.START)
            for line in iter(self._popen.stdout.readline, ''):
                if self._output is not None:
                    self._output.write(str(line))
            self._popen.stdout.close()
            self._popen.wait()
        except Exception as e:
            if self._output is not None:
                self._output.write('Process has failed: {}'.format(exception_to_string(e)))
        finally:
            self._popen = None
            if self._thread_queue is not None:
                self._thread_queue.put(GUIProvider.STOP)
            try:
                if temp_input and input_file is not None:
                    os.remove(input_file)

                input_file = None
            finally:
                if output_file is not None:
                    os.remove(output_file)
                    output_file = None
