# COLMAP-QR-Position-Detector
Compute the 3D position of a QR code in world space from a COLMAP reconstruction of a scene using RANSAC and Least Squares.


1. Load in all the images + cameras from the COLMAP dataset
2. Search each image for the QR code, if found mark down the pixel coordinates of four corners
3. For each camera/image with a QR code found, compute a ray going from the camera origin towards each corner of the QR code. Store these as observations for each corner.
4. For each corner:
    1. Perform RANSAC ->
        1. randomly select 2 observations, 
        2. use least squares to optimize 3D position for these two points (minimizes re-projection error)
        3. Identify all inliers (points with re-projection error < THRESH)
    2. With the largest set of inliers, use least squares to refine the 3D position (minimize re-projection error). 
    3. Use least squares output as the 3D position of the corner
