
Institut de Robòtica i Informàtica Industrial, CSIC-UPC. Author Irene Garcia-Camacho (igarcia@iri.upc.edu).

# Standardization of cloth objects and its relevance in robotic manipulation

Repository used in the article of the same title submitted to ICRA 2024. This respository includes the necessary packages to measure the stiffness of the cloth objects based on the drape test [1], adapted for robotic applications. It serves to measure the drape area of the cloth and the corresponding stiffness metric, independently of the camera brand or setup. It includes two scripts:

- **px_to_cm.py**: Python script to obtain the pixel to centimeter ratio for obtaining a common unit for all camera brands and setups.
- **stiffness.py**: Python script to measure the stiffness value of the garment.


## Execution

1. Follow the steps to setup the camera, aruco pattern and cloth objects and take zenithal color images.
2. Save the images in a folder in the root folder with name "data".
3. Use the image with the aruco pattern to compute the pixel to centimeter ratio:

``pyython px_to_cm.py``

4. Compute the stiffness of the cloth object through their zenithal images:

``python stiffness.py``


## Dependencies

- Python
- OpenCV

## References

[1] C.G.E., "The measurement of fabric drape", Journal of the Textile Institute, vol. 59, pp. 253-260, 1968.
