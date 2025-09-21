import os
import sys
import argparse
import numpy as np
import open3d as o3d
from utils.vis_utils import set_viewpoint_ctr, set_camera_pose
from utils.io import load_mesh, load_predicted_poses,save_camera_info


# Camera and rendering parameters
H = 320
W = 320
focal = 277
fx = focal
fy = focal
cx = W/2.0 - 0.5
cy = H/2.0 - 0.5


def capture_best_viewpoint_image(vis, waypoint_idx, x_angle, y_angle, output_folder):
    """
    Capture an image at the best viewpoint for a given waypoint.
    
    Args:
        vis: Open3D visualizer
        waypoint_idx: Index of the waypoint (for naming)
        x_angle: X-axis rotation angle in degrees
        y_angle: Y-axis rotation angle in degrees
        output_folder: Folder to save the captured image
    """
    # Capture the image
    image = vis.capture_screen_float_buffer(True)
    image = np.asarray(image)
    image = (image * 255).astype(np.uint8)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate filename
    filename = f"waypoint_{waypoint_idx + 1:05d}_x{int(x_angle)}_y{int(y_angle)}.jpg"
    filepath = os.path.join(output_folder, filename)
    
    # Save the image
    o3d.io.write_image(filepath, o3d.geometry.Image(image))
    
    print(f"Saved: {filename}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Capture images at best viewpoints for waypoints")
    
    # Arguments with default values
    parser.add_argument('--mesh-file', type=str, default='./example_data/yPKGKBCyYx8.glb',
                       help='Path to mesh file (.glb, .obj, .ply, etc.)')
    parser.add_argument('--poses-file', type=str, default='./example_data/predicted_poses.npz',
                       help='Path to predicted poses file (.npz)')
    parser.add_argument('--output-folder', type=str, default='./example_data/best_viewpoint_images',
                       help='Output folder for captured images (default: ./example_data/best_viewpoint_images)')
    
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.mesh_file):
        raise FileNotFoundError(f"Mesh file not found: {args.mesh_file}")
    if not os.path.exists(args.poses_file):
        raise FileNotFoundError(f"Poses file not found: {args.poses_file}")

    # Set up paths
    mesh_path = args.mesh_file
    poses_path = args.poses_file
    output_folder = args.output_folder
    
    try:
        # Load data
        print("Loading mesh...")
        scene_mesh = load_mesh(mesh_path)
        
        if not scene_mesh or scene_mesh.is_empty():
            raise RuntimeError(f"Failed to load mesh from {mesh_path}")
        
        print("Loading waypoints and angles...")
        waypoints, angles, extrinsics = load_predicted_poses(poses_path)
        
        # Setup visualizer
        print("Setting up visualizer...")
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=W, height=H, visible=False)
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().light_on = False
        vis.add_geometry(scene_mesh, reset_bounding_box=True)
        
        # Set viewpoint control
        ctr = set_viewpoint_ctr(vis)
        param = ctr.convert_to_pinhole_camera_parameters()
        
        # Set camera intrinsics
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        
        print(f"Starting image capture for {len(waypoints)} waypoints...")
        
        # Process each waypoint
        successful_captures = 0
        for i in range(len(waypoints)):
            waypoint = waypoints[i]
            extrinsic = extrinsics[i]
            x_angle, y_angle = angles[i]

            # Skip if angles are NaN (from a failed inference)
            if np.isnan(x_angle) or np.isnan(y_angle):
                print(f"Waypoint {i + 1}: Skipping due to invalid angles (NaN)")
                continue

            print(f"Processing waypoint {i + 1}/{len(waypoints)}: "
                  f"pos={waypoint}, angles=({x_angle:.1f}°, {y_angle:.1f}°)")
            
            # Directly set camera using extrinsic matrix
            set_camera_pose(vis, extrinsic)

            success = capture_best_viewpoint_image(
                vis, i, x_angle, y_angle, output_folder
            )
            
            if success:
                successful_captures += 1
        
        # Save camera information
        save_camera_info(waypoints, angles, output_folder)
        
        print(f"\nCompleted! Successfully captured {successful_captures}/{len(waypoints)} images")
        print(f"Images saved to: {output_folder}")
        
        # Cleanup
        vis.clear_geometries()
        vis.destroy_window()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()