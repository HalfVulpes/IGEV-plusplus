#!/usr/bin/env python
import sys
sys.path.append('core')

import rospy
import message_filters
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import torch
from igev_stereo import IGEVStereo
from utils.utils import InputPadder
from sensor_msgs import point_cloud2
import yaml
import argparse
import cv2
import torch.nn as nn

class IGEVStereoNode:
    def __init__(self, params_file):
        # Read parameters from the provided YAML file
        with open(params_file, 'r') as file:
            params = yaml.safe_load(file)

        self.fx = params['fx']
        self.fy = params['fy']
        self.cx = params['cx']
        self.cy = params['cy']
        self.baseline = params['baseline']
        self.cutoff = params.get('cutoff', 10.0)
        self.ds_voxel_size = params.get('ds_voxel_size', 0.05)

        self.extrinsic_matrix = np.array(params['extrinsic_matrix'])
        self.denoise = params.get('denoise', False)
        self.nb_neighbors = params.get('nb_neighbors', 20)
        self.std_ratio = params.get('std_ratio', 2.0)

        self.left_image_topic = params.get('left_image_topic')
        self.right_image_topic = params.get('right_image_topic')
        self.pointcloud_topic = params.get('pointcloud_topic')

        self.keyframe_rate = params.get('keyframe_rate')  # in Hz

        # Model parameters
        model_params = {
            'mixed_precision': params.get('mixed_precision', True),
            'valid_iters': params.get('valid_iters', 16),
            'hidden_dims': params.get('hidden_dims', [128, 128, 128]),
            'corr_levels': params.get('corr_levels', 2),
            'corr_radius': params.get('corr_radius', 4),
            'n_downsample': params.get('n_downsample', 2),
            'n_gru_layers': params.get('n_gru_layers', 3),
            'max_disp': params.get('max_disp', 768),
            's_disp_range': params.get('s_disp_range', 48),
            'm_disp_range': params.get('m_disp_range', 96),
            'l_disp_range': params.get('l_disp_range', 192),
            's_disp_interval': params.get('s_disp_interval', 1),
            'm_disp_interval': params.get('m_disp_interval', 2),
            'l_disp_interval': params.get('l_disp_interval', 4)
        }

        restore_ckpt = params['restore_ckpt']
        self.valid_iters = params.get('valid_iters', 16)
        self.mixed_precision = params.get('mixed_precision', True)
        model_args = argparse.Namespace(**model_params)
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = IGEVStereo(model_args)
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(restore_ckpt, map_location=self.DEVICE))
        self.model.to(self.DEVICE)
        self.model.eval()
        rospy.init_node('igev_stereo_node')
        self.bridge = CvBridge()

        self.latest_left_image = None
        self.latest_right_image = None

        self.left_sub = message_filters.Subscriber(self.left_image_topic, Image)
        self.right_sub = message_filters.Subscriber(self.right_image_topic, Image)

        ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.sync_callback)

        self.pc_pub = rospy.Publisher(self.pointcloud_topic, PointCloud2, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.keyframe_rate), self.timer_callback)

        rospy.loginfo("IGEV Stereo Node initialized.")

    def sync_callback(self, left_msg, right_msg):
        # Store the latest synchronized images
        self.latest_left_image = left_msg
        self.latest_right_image = right_msg

    def timer_callback(self, event):
        if self.latest_left_image is not None and self.latest_right_image is not None:
            try:
                self.process_images(self.latest_left_image, self.latest_right_image)
            except Exception as e:
                rospy.logerr(f"Error in timer_callback: {e}")
            finally:
                # Reset the images after processing
                self.latest_left_image = None
                self.latest_right_image = None
        else:
            rospy.logwarn("No synchronized images available for processing.")

    def process_images(self, left_msg, right_msg):
        # Convert images
        left_image_cv = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        right_image_cv = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        left_image = cv2.cvtColor(left_image_cv, cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(right_image_cv, cv2.COLOR_BGR2RGB)
        left_image_tensor = self.prepare_image(left_image)
        right_image_tensor = self.prepare_image(right_image)
        disp = self.compute_disparity(left_image_tensor, right_image_tensor)
        point_cloud = self.disparity_to_point_cloud(disp)
        point_cloud_transformed = self.apply_extrinsic_to_point_cloud(point_cloud)
        if self.denoise:
            point_cloud_transformed = self.denoise_point_cloud(point_cloud_transformed)
        distances = np.linalg.norm(point_cloud_transformed, axis=1)
        valid_indices = distances <= self.cutoff
        point_cloud_transformed = point_cloud_transformed[valid_indices]
        point_cloud_transformed = self.downsample_point_cloud(point_cloud_transformed)
        self.publish_point_cloud(point_cloud_transformed)

    def prepare_image(self, image):
        # (H, W, 3), RGB format
        # Convert to torch
        img = torch.from_numpy(image).permute(2, 0, 1).float()  # [C, H, W]
        img = img.unsqueeze(0).to(self.DEVICE)  # Add batch dimension
        return img

    def compute_disparity(self, left_image, right_image):
        # Pad images
        padder = InputPadder(left_image.shape, divis_by=32)
        left_image, right_image = padder.pad(left_image, right_image)
        with torch.no_grad():
            disp = self.model(left_image, right_image, iters=self.valid_iters, test_mode=True)
            disp = padder.unpad(disp)
        disp_numpy = disp.cpu().numpy().squeeze()
        return disp_numpy

    def disparity_to_point_cloud(self, disparity_map):
        height, width = disparity_map.shape
        u_coords = np.arange(width)
        v_coords = np.arange(height)
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)
        u = u_grid.flatten()
        v = v_grid.flatten()
        disparity = disparity_map.flatten()

        # Filter out invalid disparities
        valid = disparity > 0
        disparity = disparity[valid]
        u = u[valid]
        v = v[valid]

        z = (self.fx * self.baseline) / disparity
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        point_cloud = np.vstack((x, y, z)).T

        return point_cloud

    def apply_extrinsic_to_point_cloud(self, points):
        # Convert the point cloud to homogeneous coordinates
        ones = np.ones((points.shape[0], 1))
        homogeneous_points = np.hstack((points, ones))
        transformed_points = homogeneous_points @ self.extrinsic_matrix.T
        return transformed_points[:, :3]

    def denoise_point_cloud(self, points):
        """
        Remove outliers from point cloud using statistical outlier removal.
        """
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio)
            filtered_pcd = pcd.select_by_index(ind)
            filtered_points = np.asarray(filtered_pcd.points)

            return filtered_points
        except ImportError:
            rospy.logwarn("Open3D not installed. Skipping denoising.")
            return points

    def downsample_point_cloud(self, points):
        """
        Downsample the point cloud using voxel grid filtering.
        """
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.ds_voxel_size)
            downsampled_points = np.asarray(downsampled_pcd.points)
            return downsampled_points
        except ImportError:
            rospy.logwarn("Open3D not installed. Skipping downsampling.")
            return points

    def publish_point_cloud(self, points):
        pc_msg = self.convert_point_cloud_to_ros_msg(points)
        self.pc_pub.publish(pc_msg)

    def convert_point_cloud_to_ros_msg(self, points):
        # Create PointCloud2 message
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'camera_link'

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]

        pc_data = [tuple(p) for p in points]

        pc_msg = point_cloud2.create_cloud(header, fields, pc_data)
        return pc_msg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IGEV Stereo Node")
    parser.add_argument('--param', required=True, help='Path to the parameter YAML file.')
    args = parser.parse_args()

    try:
        node = IGEVStereoNode(args.param)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass