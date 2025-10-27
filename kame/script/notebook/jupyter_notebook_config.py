"""
Copyright 2025 ISSP, University of Tokyo, Japan.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# Picked up and renamed from those in 
# https://git.fmrib.ox.ac.uk/yqzheng1/fsleyes

import os

c.ContentsManager.untitled_notebook     = 'KAME_notebook'
c.Session.key                           = b''
#c.NotebookApp.port                      = {{ kame_nbserver_port }}
c.NotebookApp.port_retries              = 50
c.NotebookApp.token                     = os.environ.get('KAME_NOTEBOOK_SERVER_TOKEN')
c.NotebookApp.password                  = ''
#c.NotebookApp.notebook_dir              = '{{ kame_nbserver_dir }}'
#c.NotebookApp.extra_static_paths        = ['{{ kame_nbserver_static_dir }}']
c.NotebookApp.answer_yes                = True
#c.NotebookApp.extra_nbextensions_path   = ['{{ kame_nbextension_dir }}']
c.NotebookApp.kernel_manager_class      = 'notebook_kame_kernel_manager.KAMENotebookKernelManager'
#c.NotebookApp.kernel_spec_manager_class = 'kame_kernelspecmanager.kameKernelSpecManager'
# c.InteractiveShellApp.exec_lines = [
#     'print("\\nUse \'KILL\' button in KAME instead of \'interrupt kernel\'\\n")'
# ]

# inject our kernel connection
# file into the kernel manager
from notebook_kame_kernel_manager import KAMENotebookKernelManager
KAMENotebookKernelManager.connfile = os.environ.get('KAME_IPYTHON_CONNECTION_FILE')
KAMENotebookKernelManager.kame_pid = int(os.environ.get('KAME_PID'))
