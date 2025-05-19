import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull


def filter_isolated_points(point_cloud, distance_threshold=0.1):
    """ Filter isolated points from cloud that are outside a threshold

    Args:
        point_cloud (open3d point cloud): point cloud data
        distance_threshold (float, optional): distance for points to be included in final point cloud. Defaults to 0.1.

    Returns:
        open3d point cloud: filtered point cloud 
    """
    # Convert point cloud to a numpy array for easy manipulation
    points = np.asarray(point_cloud.points)
    
    # Use a KDTree for efficient neighbor search
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    
    # List to hold indices of points that should be kept
    valid_indices = []
    
    for i in range(len(points)):
        # Find neighbors within the specified radius
        [k, idx, _] = kdtree.search_radius_vector_3d(point_cloud.points[i], distance_threshold)
        
        # If more than 1 point is found (including the point itself), keep this point
        if k > 1:
            valid_indices.append(i)
    
    # Filter points and colors based on valid indices
    filtered_points = points[valid_indices]
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        filtered_colors = colors[valid_indices]
    
    # Create a new point cloud with the filtered points (and colors if available)
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    if point_cloud.has_colors():
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    return filtered_pcd

def load_pc_and_downsample(file_path, z_threshold=0, neighbor_threshold=0.1, voxel_size=0.1):
    """ Load point cloud and filter/downsample

    Args:
        file_path (str): file name/path to point cloud data
        z_threshold (int, optional): distance to include points in the z (front) direction. Defaults to 0.
        neighbor_threshold (float, optional): distance for points to be included in final point cloud. Defaults to 0.1.
        voxel_size (float, optional): size of voxel. Defaults to 0.1.

    Returns:
        open3d point cloud: filtered point cloud
    """
    pcd = o3d.io.read_point_cloud(file_path)

    points = np.asarray(pcd.points) / 1000.0 # Convert from mm to m
    colors = np.asarray(pcd.colors)

    # Invert axes (if neccesary)
    points[:, 0] *= -1  # Invert the x-axis
    points[:, 1] *= -1  # Invert the y-axis
    # points[:, 2] *= -1  # Invert the z-axis

    # Filter points and corresponding colors based on the z-coordinate (maintaining original colors)
    filtered_indices = points[:, 2] <= z_threshold
    filtered_points = points[filtered_indices]
    filtered_colors = colors[filtered_indices]

    # Create a new point cloud with the filtered points and colors
    z_filtered_pcd = o3d.geometry.PointCloud()
    z_filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    z_filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)  # Preserve the colors

    # Apply voxel downsampling (set the size of each voxel)
    downpcd = z_filtered_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Filter out any isolated neighboring points
    filtered_pcd = filter_isolated_points(downpcd, distance_threshold=neighbor_threshold)

    return filtered_pcd

def rectangular_prism_geometry(height, width, depth):
    """ Generate the geometry of a rectangular prism

    Args:
        height (float): height (y-axis)
        width (float): width (x-axis)
        depth (float): depth (z-axis)

    Returns:
        opne3d geometry: mesh of rectangular prism
    """
    prism = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    prism.translate(-prism.get_center())
    return prism

def parallelepiped_geometry(height, width, depth, theta):
    """ Generate the geometry of a parallelepiped

    Args:
        height (float): height (y-axis)
        width (float): width (x-axis)
        depth (float): depth (z-axis)
        theta (float): skew angle of prism (yz-plane)

    Returns:
        opne3d geometry: mesh of parallelepiped
    """
    # Define the rotation matrix around the x-axis
    R_x = R.from_euler('x', theta).as_matrix()

    # Define the 8 vertices of the parallelepiped, skewing in the y direction
    vertices = np.array([
        [0, 0, 0],                   # v0
        [2*width, 0, 0],             # v1
        [2*width, 2*height, 0],      # v2
        [0, 2*height, 0],            # v3
        [0, depth*np.tan(theta), depth],     # v4 (skewed by theta in y direction)
        [2*width, depth*np.tan(theta), depth],   # v5 (skewed by theta in y direction)
        [2*width, 2*height + depth*np.tan(theta), depth], # v6 (skewed by theta in y direction)
        [0, 2*height + depth*np.tan(theta), depth]    # v7 (skewed by theta in y direction)
    ])

    # Apply the rotation to each vertex
    rotated_vertices = np.dot(vertices, R_x.T)

    # Define the triangles that make up the parallelepiped
    triangles = np.array([
        [0, 1, 2], [2, 3, 0],  # Bottom face
        [4, 5, 6], [6, 7, 4],  # Top face
        [0, 1, 5], [5, 4, 0],  # Front face
        [1, 2, 6], [6, 5, 1],  # Right face
        [2, 3, 7], [7, 6, 2],  # Back face
        [3, 0, 4], [4, 7, 3]   # Left face
    ])

    # Create the mesh with rotated vertices
    parallelepiped = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(rotated_vertices),
        o3d.utility.Vector3iVector(triangles)
    )

    # Optionally, translate the parallelepiped to center it at the origin
    parallelepiped.translate(-parallelepiped.get_center())

    # Assign a uniform color to the entire parallelepiped
    color = [0.0, 0.5, 1.0]  # Cyan color
    parallelepiped.paint_uniform_color(color)

    return parallelepiped, rotated_vertices

def voxelize_shape(geometry, voxel_size, vis=False, pyb_tranform=True):
    """ Voxelize a mesh and extract the voxel center coordinates

    Args:
        geometry (open3d geometry): geometry to voxelize
        voxel_size (float): voxel size
        vis (bool, optional): visualize the voxelized geometry. Defaults to False.
        pyb_tranform (bool, optional): transform from open3d coordinates to PyBullet coordinates. Defaults to True.

    Returns:
        float lists: voxel center coordinates and their indices
    """
    geometry_type = type(geometry)

    # Voxelize the geometry with a specified voxel size
    if geometry_type == o3d.geometry.TriangleMesh:    
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(geometry, voxel_size=voxel_size)

    if geometry_type == o3d.geometry.PointCloud:
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(geometry, voxel_size=voxel_size)

    # Get the voxel data
    voxels = voxel_grid.get_voxels()
    
    # Extract voxel centers and their indices
    voxel_centers, voxel_indices = zip(*[(voxel_grid.get_voxel_center_coordinate(voxel.grid_index), voxel.grid_index) for voxel in voxels])
    
    if vis:
        # Create a coordinate frame 
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        # Visualize the voxel grid and the coordinate frame together
        o3d.visualization.draw_geometries([voxel_grid, coordinate_frame])

    if pyb_tranform:
        # Transform voxel data from open3d coordinate frame to pybullet coordinate frame
        R_x = R.from_euler('x', np.pi/2).as_matrix()
        rotated_voxel = np.dot(voxel_centers, R_x.T)
        voxel_centers = rotated_voxel

        y_trans = min(voxel_centers[:, 1])
        z_trans = min(voxel_centers[:, 2])

        # translation = np.array([0, np.abs(y_trans), np.abs(z_trans)])
        translation = np.array([0, 0, np.abs(z_trans)])
        voxel_centers += translation

    return voxel_centers, voxel_indices

def generate_parallelepiped_voxels(height, width, depth, theta, voxel_size, pyb_trans=True):
    """ Generate discrete 3D coordinates of the volume that makes up the mesh """
    # Define vertices of the parallelepiped
    vertices = np.array([
        [0, 0, 0],  # v0
        [2*width, 0, 0],  # v1
        [2*width, 2*depth, 0],  # v2
        [0, 2*depth, 0],  # v3
        [0, height*np.tan(theta), height],  # v4
        [2*width, height*np.tan(theta), height],  # v5
        [2*width, 2*depth + height*np.tan(theta), height],  # v6
        [0, 2*depth + height*np.tan(theta), height]  # v7
    ])

    # Get the axis-aligned bounding box of the parallelepiped
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)

    # Generate voxel grid coordinates
    x = np.arange(min_coords[0], max_coords[0] + voxel_size, voxel_size)
    y = np.arange(min_coords[1], max_coords[1] + voxel_size, voxel_size)
    z = np.arange(min_coords[2], max_coords[2] + voxel_size, voxel_size)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    voxel_coords = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

    # Filter voxels inside the parallelepiped
    inside_voxels = np.array([coord for coord in voxel_coords if is_point_in_parallelepiped(coord, vertices)])

    if pyb_trans:
        inside_voxels[:, 0] -= max(inside_voxels[:, 0]) / 2
        inside_voxels[:, 1] *= -1
        inside_voxels[:, 1] -= min(inside_voxels[:, 1]) / 2

    return inside_voxels

def is_point_in_parallelepiped(point, vertices):
    """ Check if a point is inside the parallelepiped defined by its vertices """
    # Create a convex hull from the vertices
    hull = ConvexHull(vertices)
    # Check if the point is inside the convex hull
    new_point = np.append(point, 1)  # Append 1 for homogeneous coordinates
    return np.all(np.dot(hull.equations[:, :-1], new_point[:-1]) <= -hull.equations[:, -1])


if __name__ == '__main__':
    # Define parameters for the pointcloud
    # file_path = "./point_clouds/pointcloud_2.ply"
    # z_threshold = 1.5 # meters
    # neighbor_threshold = 0.15 # meters
    # voxel_size = 0.1 # meters

    # rectangular_prism = rectangular_prism_geometry(height, width, depth)
    # filtered_pcd = load_pc_and_downsample(file_path, z_threshold, neighbor_threshold, voxel_size)

    # Define parameters for the parallelepiped
    voxel_size = 0.025 # meters
    vis = False

    # Define the dimensions of the parallelepiped
    width, height, depth = 0.75, 1, 0.3  # 2width x 2height x 0.5D
    theta = np.deg2rad(18.435) # Angle provided by Martin (WSU) 9/11/2024 
    # parallelepiped, vertices = parallelepiped_geometry(height, width, depth, theta)
    voxel_centers = generate_parallelepiped_voxels(height, width, depth, theta, voxel_size, pyb_trans=True)

    print(len(voxel_centers))

    # voxel_centers, voxel_indices = voxelize_shape(parallelepiped, voxel_size=voxel_size, vis=vis, fill_shape=True)

    # np.savetxt('./data/voxel_data_parallelepiped.csv', np.hstack((voxel_centers, voxel_indices)))
    np.savetxt('./data/voxel_data_parallelepiped.csv', voxel_centers)
