import numpy as np


class LoadObjects:
    def __init__(self, con, ground_plane=True) -> None:
        """ Object loader class

        Args:
            con (class): PyBullet client - an instance of the started env
        """
        self.con = con
        self.ground_plane = ground_plane
        self.load_objects()

    def load_urdf(self, urdf_name, start_pos=[0, 0, 0], start_orientation=[0, 0, 0], fix_base=True, radius=None, flags=0):
        """ Load a urdf using PyBullets urdf loader

        Args:
            urdf_name (str): filename/path to urdf file
            start_pos (list, optional): starting origin. Defaults to [0, 0, 0].
            start_orientation (list, optional): starting orientation. Defaults to [0, 0, 0].
            fix_base (bool, optional): fixed or floating. Defaults to True.
            radius (float, optional): radius of loaded object. Defaults to None.

        Returns:
            int: PyBullet object ID
        """
        orientation = self.con.getQuaternionFromEuler(start_orientation)
        if radius is None:
            objectId = self.con.loadURDF(urdf_name, start_pos, orientation, useFixedBase=fix_base, flags=flags)
        else:
            objectId = self.con.loadURDF(urdf_name, start_pos, globalScaling=radius, useFixedBase=fix_base, flags=flags)
            self.con.changeVisualShape(objectId, -1, rgbaColor=[0, 1, 0, 1]) 
        return objectId

    def load_objects(self):
        """ Load objects into the started PyBullet simulation
        """
        if self.ground_plane:
            self.planeId = self.load_urdf("plane.urdf")

            self.collision_objects = [self.planeId]
        else:
            self.collision_objects = []