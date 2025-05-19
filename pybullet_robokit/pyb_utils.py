import atexit
import os
import sys
import ctypes
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client as bc


class PybUtils:
    def __init__(self, renders: bool = False) -> None:
        """ Base class for the PyBullet Client

        Args:
            env (class): initialized class that starts this base class
            renders (bool, optional): visualize the env with the PyBullet GUI. Defaults to False.
        """
        self.renders = renders
        self.step_time = 1 / 240

        self.con = None

        self.setup_pybullet()
    
    def setup_pybullet(self) -> None:
        # New class for pybullet
        if self.renders:
            self.con = bc.BulletClient(connection_mode=p.GUI)
            self.con.configureDebugVisualizer(self.con.COV_ENABLE_GUI, 0)
        else:
            # Suppress stdout and stderr at the OS level
            # libc = ctypes.CDLL(None)
            if sys.platform.startswith('linux'):
                libc = ctypes.CDLL("libc.so.6")
            elif sys.platform == "darwin":
                libc = ctypes.CDLL("libc.dylib")
            elif sys.platform == "win32":
                libc = ctypes.CDLL("msvcrt.dll")  # Microsoft C runtime
            else:
                raise RuntimeError("Unsupported OS")
            original_stdout = os.dup(1)  # Duplicate stdout file descriptor
            original_stderr = os.dup(2)  # Duplicate stderr file descriptor

            null_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(null_fd, 1)  # Redirect stdout to /dev/null
            os.dup2(null_fd, 2)  # Redirect stderr to /dev/null
            os.close(null_fd)

            try:
                self.con = bc.BulletClient(connection_mode=p.DIRECT)
            finally:
                os.dup2(original_stdout, 1)  # Restore stdout
                os.dup2(original_stderr, 2)  # Restore stderr
                os.close(original_stdout)
                os.close(original_stderr)

        self.con.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.con.setTimeStep(self.step_time)

        self.enable_gravity()

        self.con.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=90, cameraPitch=-10,
                                            cameraTargetPosition=[0, 0.75, 0.75])
        
        atexit.register(self.disconnect)

    def disable_gravity(self):
        self.con.setGravity(0, 0, 0)

    def enable_gravity(self):
        self.con.setGravity(0, 0, -9.81)

    def disconnect(self):
        self.con.disconnect()