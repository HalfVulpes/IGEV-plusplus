import numpy as np
import argparse
import os
import yaml
import re
import open3d as o3d

def read_disparity_map(filepath):
    return np.loadtxt(filepath)

def disparity_to_point_cloud(disparity_map, fx, fy, cx, cy, baseline):
    height, width = disparity_map.shape

    u_coords = np.arange(width)
    v_coords = np.arange(height)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    u = u_grid.flatten()
    v = v_grid.flatten()
    disparity = disparity_map.flatten()

    # Filter out invalid disparities (disparity <= 0)
    valid = disparity > 0
    disparity = disparity[valid]
    u = u[valid]
    v = v[valid]

    z = (fx * baseline) / disparity
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    point_cloud = np.vstack((x, y, z)).T

    return point_cloud

def apply_extrinsic_to_point_cloud(points, extrinsic_matrix):
    # Convert the point cloud to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    homogeneous_points = np.hstack((points, ones))
    transformed_points = homogeneous_points @ extrinsic_matrix.T
    return transformed_points[:, :3]

def denoise_point_cloud(points, nb_neighbors=20, std_ratio=2.0):
    """
    Remove outliers from point cloud using statistical outlier removal.

    Parameters:
    - points: numpy array of shape (N, 3)
    - nb_neighbors: int, number of neighbors to consider for each point
    - std_ratio: float, threshold ratio

    Returns:
    - filtered_points: numpy array of shape (M, 3), where M <= N
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    filtered_pcd = pcd.select_by_index(ind)
    filtered_points = np.asarray(filtered_pcd.points)

    return filtered_points

def save_point_cloud_ascii(points, filename):
    with open(filename, 'w') as ply_file:
        # Write the PLY header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(points)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")

        # Use NumPy to write all points at once
        np.savetxt(ply_file, points, fmt='%f %f %f')

    print(f"Point cloud saved to {filename}")

def load_extrinsic_from_yaml(yaml_file, camera_id='body_T_cam0'):
    with open(yaml_file, 'r') as file:
        yaml_content = yaml.safe_load(file)

    if camera_id not in yaml_content:
        raise ValueError(f"Camera ID '{camera_id}' not found in YAML file.")

    extrinsic = yaml_content[camera_id]

    if 'rows' in extrinsic and 'cols' in extrinsic and 'data' in extrinsic:
        rows = extrinsic['rows']
        cols = extrinsic['cols']
        data = extrinsic['data']
        extrinsic_matrix = np.array(data).reshape((rows, cols))
    else:
        raise ValueError("Extrinsic matrix is not in expected OpenCV format.")

    return extrinsic_matrix

def main():
    parser = argparse.ArgumentParser(description="Convert disparity maps to point clouds.")
    parser.add_argument('--in', '--input_folder', dest='input_folder', default='output', help='Input folder containing disparity maps (default: output)')
    parser.add_argument('--out', '--output_folder', dest='output_folder', default='output_pointcloud', help='Output folder to save point clouds (default: output_pointcloud)')
    parser.add_argument('--fx', type=float, default=425.99684953503163, help='Focal length x (default: 425.99684953503163)')
    parser.add_argument('--fy', type=float, default=426.0108446650122, help='Focal length y (default: 426.0108446650122)')
    parser.add_argument('--cx', type=float, default=426.5960073761994, help='Optical center x (default: 426.5960073761994)')
    parser.add_argument('--cy', type=float, default=240.4590369784203, help='Optical center y (default: 240.4590369784203)')
    parser.add_argument('--baseline', type=float, default=0.05000244116935238, help='Baseline between the stereo cameras (default: 0.05000244116935238)')
    parser.add_argument('--extrinsic_yaml_file', type=str, help='YAML file containing the extrinsic matrices')
    parser.add_argument('--camera_id', type=str, default='body_T_cam0', help='Camera ID to use from YAML file (default: body_T_cam0)')
    parser.add_argument('--denoise', default=True, action='store_true', help='Apply denoising to the point clouds')
    parser.add_argument('--nb_neighbors', type=int, default=20, help='Number of neighbors to consider for denoising (default: 20)')
    parser.add_argument('--std_ratio', type=float, default=2.0, help='Standard deviation ratio for denoising (default: 2.0)')
    parser.add_argument('--ds_voxel_size', type=float, default=0.05, help='Downsampled resolution')
    parser.add_argument('--cutoff', type=float, default=10.0, help='Cut-off distance in meters to filter points beyond this distance (default: 10.0)')
    parser.add_argument('--start_id', type=int, default=0, help='The beginning index id of the point cloud files')
    parser.add_argument('--end_id', type=int, default=100, help='The ending index id of the point cloud files')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    fx = args.fx
    fy = args.fy
    cx = args.cx
    cy = args.cy
    baseline = args.baseline

    if args.extrinsic_yaml_file:
        extrinsic_matrix = load_extrinsic_from_yaml(args.extrinsic_yaml_file, args.camera_id)
    else:
        extrinsic_matrix = np.array([
            [0.00137718, -0.02366516,  0.99971899,  0.13932612],
            [-0.99998467,  0.00532906,  0.00150369,  0.015785],
            [-0.00536315, -0.99970574, -0.02365746,  0.00489279],
            [0, 0, 0, 1]
        ])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith('.dspm')]
    files.sort()
    files = files[args.start_id:args.end_id]

    for idx, filename in enumerate(files):
        disparity_map_file = os.path.join(input_folder, filename)
        disparity_map = read_disparity_map(disparity_map_file)
        point_cloud = disparity_to_point_cloud(disparity_map, fx, fy, cx, cy, baseline)
        point_cloud_transformed = apply_extrinsic_to_point_cloud(point_cloud, extrinsic_matrix)

        if args.denoise:
            point_cloud_transformed = denoise_point_cloud(
                point_cloud_transformed,
                nb_neighbors=args.nb_neighbors,
                std_ratio=args.std_ratio
            )

        distances = np.linalg.norm(point_cloud_transformed, axis=1)
        valid_indices = distances <= args.cutoff
        point_cloud_transformed = point_cloud_transformed[valid_indices]

        match = re.search(r'\d+', filename)
        if match:
            identifier = match.group()
        else:
            identifier = f"{idx:06d}"

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_transformed)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=args.ds_voxel_size)

        output_filename = f"point_cloud_{identifier}.ply"
        output_file_path = os.path.join(output_folder, output_filename)
        save_point_cloud_ascii(np.asarray(downsampled_pcd.points), output_file_path)

if __name__ == "__main__":
    main()