import numpy as np
import os
import cv2
import pycolmap
from scipy.optimize import least_squares

COLMAP_PATH = "../Long0.5xQR/sparse/0"
IMAGE_FOLDER = "../Long0.5xQR/images"

RANSAC_ITERS = 150
THRESH = 2.0


def load_colmap_data():
    return pycolmap.Reconstruction(COLMAP_PATH)

# returns the 4 corners of qr code in image coordinates, if QR code is present
# order is lop left, top right, bottom right, bottom left 
def find_QR_corners(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    QRDetector = cv2.QRCodeDetector()

    success, points = QRDetector.detect(img)
    if not success or points is None:
        return None

    print(f"Found QR code in {image_path}")

    # reshape to (4, 2)
    points = np.squeeze(points).astype(np.float32)
    return points


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
    reconstruction = load_colmap_data()
    print("Loaded COLMAP data")

    # store a bunch of observations of each corner for 
    corner_observations = {i: [] for i in range(4)}

    for img_id, img_info in reconstruction.images.items():
        img_path = f"{IMAGE_FOLDER}/{img_info.name}"
        
        if (not os.path.exists(img_path)):
            continue

        corners = find_QR_corners(img_path)
        if corners is None:
            continue 


        camera = reconstruction.cameras[img_info.camera_id]
        # intrinsic camera matrix -> converts 3d points in camera coords to 2d pixel coords
        cam_from_world = img_info.cam_from_world
        K = get_intrinsic_matrix(camera)
        Kinv = np.linalg.inv(K)
       
        if (not img_info.has_pose):
            print(f"Skipping image {img_info.name} because there is no valid pose")
            continue

        for corner_index, pixel in enumerate(corners):
            u,v = pixel
            p_homogenous = np.array([u, v, 1.0], dtype=np.float64)

            # see here https://github.com/colmap/colmap/issues/1596
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation
            ray_dir = R.T @ (Kinv @ p_homogenous)  
            ray_origin = - R.T @ t  

           # print(f"ray dir: {ray_dir}, ray_origin: {ray_origin}")

            corner_observations[corner_index].append({
                "camera":camera,
                "img_info": img_info,
                "pixel":pixel,
                "ray_origin": ray_origin,
                "ray_dir": ray_dir,
                })

    corner_positions = {}
    for corner_index in range(4):
        observations = corner_observations[corner_index]
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
                errors.append(np.linalg.norm(proj - obs["pixel"]))

            inliers = [obs for obs,err in zip(observations, errors) if err < THRESH]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers

        if len(best_inliers) > 0:
            corner_positions[corner_index] = triangulate_corner(best_inliers)
        else:
            print(f"Failed to triangulate corner {corner_index}")


    print("Triangulated QR code corners:")
    for index, pos in corner_positions.items():
        print(f"Corner {index}: {pos}")


if __name__ == "__main__":
    main()
