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
import os
import signal
import threading
import time

from jupyter_server.services.kernels.kernelmanager import MappingKernelManager


def parent_alive(pid):
    """Does process `pid` still exist?

    Deliberately NOT os.kill(pid, 0) on Windows: there, Python maps every
    signal other than CTRL_C_EVENT/CTRL_BREAK_EVENT onto TerminateProcess, so
    the "probe" would kill KAME outright.
    """
    if os.name == 'nt':
        import ctypes
        SYNCHRONIZE, WAIT_OBJECT_0 = 0x00100000, 0
        h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not h:
            return False
        try:
            #Signalled means the process has exited; still-running waits out
            #the (zero) timeout and reports WAIT_TIMEOUT instead.
            return ctypes.windll.kernel32.WaitForSingleObject(h, 0) != WAIT_OBJECT_0
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True     #exists, just not ours to signal
    return True


def start_kame_parent_watchdog(pid, interval=2.0, grace=10.0):
    """Shut this notebook server down when the KAME that launched it goes away.

    KAME stops the server on a clean exit, but a crash or a kill runs no code
    in KAME at all -- which is how seven orphaned servers once accumulated in
    eleven hours, each holding its port so the next launch climbed to 8889,
    8890, ...  Only a check living in THIS process survives that, and there is
    nothing to preserve: every notebook this server hands out is wired to
    KAME's embedded kernel, so without KAME it can serve nobody.

    PID reuse could in principle make this fire against an unrelated process.
    The cost of that is a server nobody could have used exiting early, so the
    simple check is the right one.
    """
    def _run():
        while parent_alive(pid):
            time.sleep(interval)
        #Signal ourselves rather than _exit: the server's own SIGTERM handler
        #shuts the kernels it started down too, which _exit would orphan.
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(grace)
        os._exit(1)

    threading.Thread(target=_run, name='kame-parent-watchdog',
                     daemon=True).start()
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


    async def interrupt_kernel(self, kernel_id, **kwargs):
        """Overrides ``MappingKernelManager.interrupt_kernel``.
        Sends SIGINT to the KAME process (not the dummy subprocess) so that
        Jupyter's stop button actually interrupts the embedded IPython kernel.
        """
        import os, signal
        os.kill(self.kame_pid, signal.SIGINT)

    async def restart_kernel(self, kernel_id, **kwargs):
        """Overrides ``MappingKernelManager.restart_kernel``.
        Skips the actual restart (which would reset ports and break the
        connection to the embedded KAME kernel) and re-patches the connection
        to keep pointing at the KAME kernel.
        """
        kernel = self._kernels.get(kernel_id)
        if kernel:
            self.__patch_connection(kernel)

