<div align="center">
  <h1 align="center">Standardization of Cloth Objects and its Relevance in Robotic Manipulation</h1>
  <p>
    <a href="http://www.iri.upc.edu/groups/perception/#ClothStandardization">
      <img src="https://img.shields.io/badge/Website-grey">
    </a>
    <a href="https://arxiv.org/abs/2403.04608">
      <img src="https://img.shields.io/badge/Arxiv-2403.04608-red">
    </a>
    <a href="https://2024.ieee-icra.org/">
      <img src="https://img.shields.io/badge/Conference-ICRA 2024-green">
    </a>
  </p>
</div>

<p align="center">
 <a href="http://www.iri.upc.edu/groups/perception/#ClothStandardization">
  <img width="360" src="1_radar_chart_final.png?raw=true" alt="Radar chart" />
 </a>
 <br>
</p>

This repository contains the code for [Standardization of Cloth Objects and its Relevance in Robotic Manipulation](http://www.iri.upc.edu/groups/perception/#ClothStandardization), with the [corresponding paper](https://arxiv.org/abs/2403.04608) accepted at the [2024 IEEE International Conference on Robotics and Automation (ICRA 2024)](https://2024.ieee-icra.org/) in Yokohama. 


Contact: Irene Garcia-Camacho (igarcia@iri.upc.edu)

## Getting Started

The respository includes the necessary packages to measure the stiffness of cloth objects based on the drape test [1], adapted to robotic applications, and independently of the camera brand or setup. The package has the following structure:

- **/data** includes the database with zenithal photos of the draped clothes, the resulting images and the stiffness results in CSV files.
- **/src** contains the necessary scripts to compute the stiffness:
    - `stiffness.py` Script to measure the stiffness value of the garment.
    - `trackbars.py` Script to obtain segmentation thresholds.


## Execution

1. Follow the steps to setup the camera, aruco pattern and cloth objects and take zenithal color images.
2. Save the images in a folder with name "data".
3. Compute the stiffness of the cloth object through its zenithal image. You will need to introduce the aruco image file (-a), the cloth image file (-i), plate diamter used (-p) and cloth dimensions (-s).

```
python3 src/stiffness.py -a <aruco_image> -i <cloth_image> -p <plate_diam> -s <short_edge_length> <long_edge_length>
```

If necessary, use before the `trackbars.py` script to obtain a better segmentation by sliding the threshold trackbars until the contour of the drapped cloth is correctly detected. Use the obtained values in the `stiffness.py` file as t_lower and t_upper values.

```
python3 src/trackbars.py -i EOS/black_flowers_v.jpg
```

4. Repeat step 3 for each garment. The resulting stiffness values will be saved on the `stiffness_data.csv` file, along with other useful information.

### Usage example

Example for measuring the stiffness of a cloth object from the Elastic Object Set (EOS) with dimensions 17x23cm, using a plate of 10cm diameter:

```
python3 src/stiffness.py -a EOS/aruco.jpg -i EOS/black_flowers_v.jpg -p 10 -s 17 23
```

<!-- ## Terminal output

The previous command will provide the drape ratio percentage (rigidity) through terminal in green, as well as some useful information. -->

## Dependencies

- Python3
- OpenCV
- CSV

## References

[1] C.G.E., "The measurement of fabric drape", Journal of the Textile Institute, vol. 59, pp. 253-260, 1968.
