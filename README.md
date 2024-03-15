<div align="center">
  <h1 align="center">Standardization of Cloth Objects and its Relevance in Robotic Manipulation</h1>
  <p>
    <a href="https://iros2022.org">
      <img src="https://img.shields.io/badge/Website-grey">
    </a>
    <a href="https://arxiv.org/abs/2403.04608">
      <img src="https://img.shields.io/badge/Arxiv-2208.10552-red">
    </a>
    <a href="https://2024.ieee-icra.org/">
      <img src="https://img.shields.io/badge/Conference-ICRA 2024-green">
    </a>
  </p>
</div>

<p align="center">
 <a href="http://www.iri.upc.edu/groups/perception/#ClothStandardization">
  <img width="360" src="imgs/1_radar_chart_final.png?raw=true" alt="Radar chart" />
 </a>
 <br>
</p>

This repository contains the code for [Standardization of Cloth Objects and its Relevance in Robotic Manipulation](http://www.iri.upc.edu/groups/perception/#ClothStandardization), with the [corresponding paper](https://arxiv.org/abs/2403.04608) accepted at the [2024 IEEE International Conference on Robotics and Automation (ICRA 2024)](https://2024.ieee-icra.org/) in Yokohama. 


Contact: Irene Garcia-Camacho (igarcia@iri.upc.edu)

## Getting Started

The respository includes the necessary packages to measure the stiffness of the cloth objects based on the drape test [1], adapted for robotic applications, and independently of the camera brand or setup. The package has the following structure:

- `**/data**` includes the database with the photos of the draped clothes, the resulting images and the stiffness results in a CSV file.
- `**/src**` contains the necessary scripts to compute the stiffness.
    - `px_to_cm.py` Script to obtain the pixel to centimeter ratio for obtaining a common unit for all camera brands and setups.
    - `stiffness.py` Script to measure the stiffness value of the garment.
    - `trackbars.py` Script to obtain segmentation thresholds.


## Execution

1. Follow the steps to setup the camera, aruco pattern and cloth objects and take zenithal color images.
2. Save the images in a folder in the root folder with name "data".
3. Use the image with the aruco pattern to compute the pixel to centimeter ratio:

```
python px_to_cm.py
```

4. Compute the stiffness of the cloth object through their zenithal images. You will need to introduce the input file name (-i), plate diamter used (-p) and cloth dimensions (-s)

```
python stiffness.py -i <file_name> -p <plate_diam> -s <short_edge_length> <long_edge_length>
```

If necessary, use the `trackbars.py` script to obtain a better segmentation by sliding the threshold trackbars until the contour of the drapped cloth is correctly detected.

5. Repeat step 4 for each garment. The resulting stiffness values will be saved on the `stiffness_data.csv` file.

### Usage example

Example for measuring the stiffness of the flat towel with dimensions 50x90cm, using a plate of 27cm diameter:

```
python stiffness.py -i sm_towel -p 28 -s 30 50
```

<!-- ## Terminal output

The previous command will provide the drape ratio percentage (rigidity) through terminal in green, as well as some useful information. -->

## Dependencies

- Python
- OpenCV
- CSV

## References

[1] C.G.E., "The measurement of fabric drape", Journal of the Textile Institute, vol. 59, pp. 253-260, 1968.
