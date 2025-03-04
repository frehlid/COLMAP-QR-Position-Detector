import numpy as np
import os
import cv2
import pycolmap
import sys
from scipy.optimize import least_squares

COLMAP_PATH = "../../Long0.5xQR/sparse/0"
IMAGE_FOLDER = "../../Long0.5xQR/images"

RANSAC_ITERS = 150
THRESH = 1.0

QR_CODE_REAL_SIZE_M = 0.204
QR_TL_REAL_HEIGHT_M = 1.55

VALID_QRs = ["306A_Wall_1", "306A_Wall_2"]

def load_colmap_data(col_path):
    return pycolmap.Reconstruction(col_path)

# returns the 4 corners of qr code in image coordinates, if QR code is present
# order is lop left, top right, bottom right, bottom left 
def find_QR_corners(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    QRDetector = cv2.QRCodeDetector()

    success, decoded_info, points, _ = QRDetector.detectAndDecodeMulti(img)
    if not success or points is None:
        return None


    qr_results = []
    for qr_string, qr_corners in zip(decoded_info, points):
        print(f"Found QR code {qr_string} in {image_path}")
        if (qr_string and qr_string in VALID_QRs):
            # reshape to (4,2)
            qr_corners = np.squeeze(qr_corners).astype(np.float32)
            qr_results.append((qr_string, qr_corners))

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


    for qr_string, corners in qr_positions.items():
        print(f"Triangulated corners for QR code: {qr_string}")

        for index, pos in corners.items():
            print(f"Corner {index}: {pos}")

        top_dist = np.linalg.norm(corners[1] - corners[0])
        scale = QR_CODE_REAL_SIZE_M / top_dist;
        print(f"Real size / calculated size: {scale}")

        gs_up = corners[0] - corners[3]
        gs_up = gs_up / np.linalg.norm(gs_up)
        gs_up *= -1

        gs_floor = corners[0] + (QR_CODE_REAL_SIZE_M / scale) * gs_up

        print(f"Position of floor {gs_floor}")

        print("")


if __name__ == "__main__":
    main()
