import numpy as np
import os
import cv2
import pycolmap
import sys
import json
from scipy.optimize import least_squares
from pyzbar.pyzbar import decode
from scipy.spatial.transform import Rotation as R

COLMAP_PATH = "../../Long0.5xQR/sparse/0"
IMAGE_FOLDER = "../../Long0.5xQR/images"

RANSAC_ITERS = 150
THRESH = 1.0

QR_CODE_REAL_SIZE_M = [0.189, 0.189, 0.189]
QR_TL_REAL_HEIGHT_M = 1.385 + QR_CODE_REAL_SIZE_M[0]

VALID_QRs = ["306A_Wall_1", "306A_Wall_2", "306A_Wall_3"]

def save_positions_to_json(qr_positions, floor, output_file):
    qr_data = {
            "floor": floor.tolist(),
            "qr_positions" : {
                qr_string: {str(index): pos.tolist() for index, pos in corners.items()}
                for qr_string, corners in qr_positions.items()
            }
    }

    with open(output_file, "w") as f:
        json.dump(qr_data, f, indent=4)

    print(f"Saved QR positions to {output_file}")

def load_colmap_data(col_path):
    return pycolmap.Reconstruction(col_path)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# returns the 4 corners of qr code in image coordinates, if QR code is present
# order is lop left, top right, bottom right, bottom left 
def find_QR_corners(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    decoded_objs = decode(img)
    if not decoded_objs:
        return None

    qr_results = []
    for obj in decoded_objs:
        qr_string = obj.data.decode("utf-8")
        if qr_string and qr_string in VALID_QRs:
            points = obj.polygon
            pts = np.array([[p.x, p.y] for p in points], dtype=np.float32)
            if pts.shape[0] != 4:
                # If we don't have exactly four points, approximate a quadrilateral
                hull = cv2.convexHull(pts)
                if hull.shape[0] == 4:
                    pts = hull.reshape((4, 2))
                else:
                    continue
            pts = order_points(pts)
            print(f"Found QR code {qr_string} in {image_path}")
            qr_results.append((qr_string, pts))
    return qr_results



def residual(x, observations):
    errors = []
    for obs in observations:
        x_cam = obs["img_info"].project_point(x)
        if (x_cam is None):
            errors.extend([9999999.0, 9999999.0])
            continue

    
        errors.extend(x_cam - obs["pixel"])
    return np.array(errors).ravel()


def get_intrinsic_matrix(camera):
    if camera.model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
        f, cx, cy = camera.params
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]], dtype=np.float64)
    elif camera.model == pycolmap.CameraModelId.PINHOLE:
        fx, fy, cx, cy = camera.params
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)
    else: 
        print(f"Camera Model {camera.model} not supported")
        return None
    return K
                     

# using multiple observations of the QR code corners, triangulate the 3D positions
def triangulate_corner(corners):
    
    # start by guessing the weighted average of all the ray intersections
    inital_guess = np.mean([o['ray_origin'] + o['ray_dir']*5 for o in corners], axis=0)

    
    # optimizing the 3d position of the QR code by minimizing error when reprojecting
    result = least_squares(residual, inital_guess, args=(corners,))
    return result.x # final 3d position


def compute_rotations(qr_positions):
    candidate_rotations = []

    for qr_string, corners in qr_positions.items():
        if not all(index in corners for index in [0, 1, 3]):
            print(f"QR code {qr_string} does not have all required corners for rotation estimation")
            continue
        
        if qr_string == VALID_QRs[2]:
            local_z = corners[1] - corners[0]
            local_z /= np.linalg.norm(local_z)

            local_y = corners[3] - corners[0]
            local_y /= np.linalg.norm(local_y)

            # estimate with cross product
            local_x = np.cross(local_y, local_z)
            local_x /= np.linalg.norm(local_x)

            R_local = np.column_stack((local_x, local_y, local_z))
            R_candidate = R_local.T # this brings scene into unity world frame (unit axes)
        else:
            local_x = corners[1] - corners[0]
            local_x /= np.linalg.norm(local_x)

            local_y = corners[3] - corners[0]
            local_y /= np.linalg.norm(local_y)

            local_z = np.cross(local_x, local_y)
            local_z /= np.linalg.norm(local_z)

            R_local = np.column_stack((local_x, local_y, local_z))
            R_candidate = R_local.T # this brings scene into unity world frame (unit axes)
        
        candidate_rotations.append(R.from_matrix(R_candidate))
    
    # average for robustness!
    if candidate_rotations:
        avg_rotation = R.mean(candidate_rotations)
        avg_rotation_matrix = avg_rotation.as_matrix()
        print("Averaged Rotation Matrix for transforming scene to Unity coordinates:")
        print(avg_rotation_matrix)



def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <COLMAP_PATH> <IMAGE_FOLDER>")
        sys.exit(1)

    COLMAP_PATH = sys.argv[1]
    IMAGE_FOLDER = sys.argv[2]

    reconstruction = load_colmap_data(COLMAP_PATH)
    print("Loaded COLMAP data")

    qr_observations = {}

    for img_id, img_info in reconstruction.images.items():
        img_path = f"{IMAGE_FOLDER}/{img_info.name}"
        
        if (not os.path.exists(img_path)):
            continue

        qr_codes = find_QR_corners(img_path)
        if qr_codes is None:
            continue 


        camera = reconstruction.cameras[img_info.camera_id]
        # intrinsic camera matrix -> converts 3d points in camera coords to 2d pixel coords
        cam_from_world = img_info.cam_from_world
        K = get_intrinsic_matrix(camera)
        Kinv = np.linalg.inv(K)
       
        if (not img_info.has_pose):
            print(f"Skipping image {img_info.name} because there is no valid pose")
            continue
        
        for qr_string, corners in qr_codes:
            if qr_string not in qr_observations:
                qr_observations[qr_string] = {i: [] for i in range(4)}

            for corner_index, pixel in enumerate(corners):
                u,v = pixel
                p_homogenous = np.array([u, v, 1.0], dtype=np.float64)

                # see here https://github.com/colmap/colmap/issues/1596
                R = cam_from_world.rotation.matrix()
                t = cam_from_world.translation
                ray_dir = R.T @ (Kinv @ p_homogenous)  
                ray_origin = - R.T @ t  

               # print(f"ray dir: {ray_dir}, ray_origin: {ray_origin}")

                qr_observations[qr_string][corner_index].append({
                   "camera": camera,
                   "img_info": img_info,
                   "pixel": pixel,
                   "ray_origin": ray_origin,
                   "ray_dir": ray_dir,
                })

    qr_positions = {}

    for qr_string, corners, in qr_observations.items():
        corner_positions = {}

        for corner_index in range(4):
            observations = corners[corner_index]
            if len(observations) < 2:
                ## TODO should I throw an error here?
                print(f"Corner {corner_index} does not have enough observations...")
                return 

            # super simple ransac -- TODO: use scikitlearn RANSAC??
            # basically just trying to remove any outlier observations before least squares is run on inlier set
            best_inliers = []
            best_err = float('inf')

            for _ in range(RANSAC_ITERS):
                sample = np.random.choice(observations, size=2, replace=False)
                pos = triangulate_corner(sample)
        
                errors = []
                for obs in observations:
                    proj = obs["img_info"].project_point(pos)
                    if (proj is None): 
                        errors.append(9999999.0)
                    else:
                        errors.append(np.linalg.norm(proj - obs["pixel"]))

                inliers = [obs for obs,err in zip(observations, errors) if err < THRESH]

                if len(inliers) > len(best_inliers):
                    best_inliers = inliers

            if len(best_inliers) > 0:
                print(f"Triangulated corner {corner_index} with {len(best_inliers)} inliers")
                corner_positions[corner_index] = triangulate_corner(best_inliers)
            else:
                print(f"Failed to triangulate corner {corner_index}")
        if len(corner_positions) == 4:
            qr_positions[qr_string] = corner_positions


    scale_estimates = {'x': [], 'y': [], 'z': []}
    gs_floor = corners[0]
    for qr_string, corners in qr_positions.items():
        print(f"Triangulated corners for QR code: {qr_string}")

        for index, pos in corners.items():
            print(f"Corner {index}: {pos}")

        edge_h = corners[1] - corners[0]
        edge_v = corners[3] - corners[0]

        if qr_string in [VALID_QRs[0], VALID_QRs[1]]:
            if abs(edge_h[0]) > 1e-6:
                scale_x_qr = QR_CODE_REAL_SIZE_M[0] / abs(edge_h[0])
                scale_estimates['x'].append(scale_x_qr)
                print(f"Scale x for {qr_string}: {scale_x_qr}")

            if abs(edge_v[1]) > 1e-6:
                scale_y_qr = QR_CODE_REAL_SIZE_M[1] / abs(edge_v[1])
                scale_estimates['y'].append(scale_y_qr)
                print(f"Scale y for {qr_string}: {scale_y_qr}")

        # the last qr is perpendicular so we handle differently
        elif qr_string == VALID_QRs[2]:
            if abs(edge_h[2]) > 1e-6:
                scale_z_qr = QR_CODE_REAL_SIZE_M[2] / abs(edge_h[2])
                scale_estimates['z'].append(scale_z_qr)
                print(f"Scale z for {qr_string}: {scale_z_qr}")

            # compute floor from this one (easier to measure in physical space)
            gs_up = corners[0] - corners[3]
            gs_up = gs_up / np.linalg.norm(gs_up)
            gs_up *= -1
            gs_floor = corners[0] + (QR_TL_REAL_HEIGHT_M / scale_z_qr) * gs_up
            print(f"Position of floor {gs_floor}")


        print("")
    
    compute_rotations(qr_positions=qr_positions)

    save_positions_to_json(qr_positions, gs_floor, "qr_positions.json")


if __name__ == "__main__":
    main()
