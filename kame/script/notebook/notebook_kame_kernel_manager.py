# -*- coding: utf-8 -*-
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
from jupyter_server.services.kernels.kernelmanager import MappingKernelManager
class KAMENotebookKernelManager(MappingKernelManager):
    """Custom jupter ``MappingKernelManager`` which forces every notebook
    to connect to the embedded KAME IPython kernel.

    See https://github.com/ebanner/extipy
    """


    connfile = ''
    """Path to the IPython kernel connection file that all notebooks should
    connect to.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # def kame_echoback():
        #     import time
        #     import sys
        #     time.sleep(10)
        #     while True:
        #         print('ping....')
        #         sys.stderr.write('pong...\n')
        #         sys.stderr.flush()
        #         sys.stdout.flush()
        #         time.sleep(2)
        #         for line in sys.stdin:
        #             print(line, end="")
        #             sys.stdout.flush()
        # import threading
        # self.thread = threading.Thread(daemon=True, target=kame_echoback)
        # self.thread.start()

    def __patch_connection(self, kernel):
        """Connects the given kernel to the IPython kernel specified by
        ``connfile``.
        """
        kernel.hb_port      = 0
        kernel.shell_port   = 0
        kernel.stdin_port   = 0
        kernel.iopub_port   = 0
        kernel.control_port = 0
        kernel.load_connection_file(self.connfile)


    async def start_kernel(self, **kwargs):
        """Overrides ``MappingKernelManager.start_kernel``. Connects
        all new kernels to the IPython kernel specified by ``connfile``.
        """
        kid    = await super().start_kernel(**kwargs)
        kernel = self._kernels[kid]
        self.__patch_connection(kernel)
        return kid


    def restart_kernel(self, *args, **kwargs):
        """Overrides ``MappingKernelManager.restart_kernel``. Does nothing. """
        pass
