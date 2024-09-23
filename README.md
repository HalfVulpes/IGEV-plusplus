## **My IGEV-plusplus**

-----

Credit goes to:

[IGEV++: Iterative Multi-range Geometry Encoding Volumes for Stereo Matching](https://arxiv.org/pdf/2409.00638) 

Gangwei Xu, Xianqi Wang, Zhaoxing Zhang, Junda Cheng, Chunyuan Liao, Xin Yang




## Citation

If you find our works useful in your research, please consider citing our papers:

```bibtex

@article{xu2024igev++,
  title={IGEV++: Iterative Multi-range Geometry Encoding Volumes for Stereo Matching},
  author={Xu, Gangwei and Wang, Xianqi and Zhang, Zhaoxing and Cheng, Junda and Liao, Chunyuan and Yang, Xin},
  journal={arXiv preprint arXiv:2409.00638},
  year={2024}
}

@inproceedings{xu2023iterative,
  title={Iterative Geometry Encoding Volume for Stereo Matching},
  author={Xu, Gangwei and Wang, Xianqi and Ding, Xiaohuan and Yang, Xin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21919--21928},
  year={2023}
}
```



## Environment

- python 3.8
- torch 1.12.1+cu113



## Dependencies

```
pip install jinja2
pip install jupyter-client pyzmq
pip install notebook jupyterlab
pip install pyyaml
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tqdm
pip install scipy
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install timm==0.5.4
```



## Usage

### extract_disparity.py:

#### Command-Line Arguments

The script accepts the following command-line arguments:

#### General Arguments

- `--restore_ckpt`: Path to the model checkpoint file.
  - **Default**: `checkpoints/sceneflow.pth`
- `--save_numpy`: If set, the script will save the disparity maps as `.npy` NumPy array files.
  - **Default**: Not set (disabled)
- `-l`, `--left_imgs`: Path to the directory containing left images.
  - **Default**: `left/`
- `-r`, `--right_imgs`: Path to the directory containing right images.
  - **Default**: `right/`
- `--output_directory`: Directory where output files will be saved.
  - **Default**: `output`
- `--mixed_precision`: Use mixed precision for faster inference.
  - **Default**: `True`
- `--valid_iters`: Number of iterations during the forward pass.
  - **Default**: `16`

#### Model Architecture Arguments

- `--hidden_dims`: Hidden state and context dimensions.
  - **Default**: `[128, 128, 128]`
- `--corr_levels`: Number of levels in the correlation pyramid.
  - **Default**: `2`
- `--corr_radius`: Width of the correlation pyramid.
  - **Default**: `4`
- `--n_downsample`: Resolution of the disparity field (1/2<sup>K</sup>).
  - **Default**: `2`
- `--n_gru_layers`: Number of hidden GRU levels.
  - **Default**: `3`
- `--max_disp`: Maximum disparity range.
  - **Default**: `768`
- `--s_disp_range`: Max disparity of small disparity-range geometry encoding volume.
  - **Default**: `48`
- `--m_disp_range`: Max disparity of medium disparity-range geometry encoding volume.
  - **Default**: `96`
- `--l_disp_range`: Max disparity of large disparity-range geometry encoding volume.
  - **Default**: `192`
- `--s_disp_interval`: Disparity interval of small disparity-range geometry encoding volume.
  - **Default**: `1`
- `--m_disp_interval`: Disparity interval of medium disparity-range geometry encoding volume.
  - **Default**: `2`
- `--l_disp_interval`: Disparity interval of large disparity-range geometry encoding volume.
  - **Default**: `4`



#### Output Files

For each image pair, the script generates the following files in the `output/` directory:

1. **Disparity Map as PNG**
   - **Filename**: `<imagename>.png`
   - **Description**: A colorized disparity map saved as a PNG image using the 'jet' colormap.
2. **Disparity Map as DSPM**
   - **Filename**: `<imagename>.dspm`
   - **Description**: The disparity map saved in ASCII format, which can be used for further processing.
3. **Disparity Map as NumPy Array (Optional)**
   - **Filename**: `<imagename>.npy`
   - **Description**: If `--save_numpy` is specified, the disparity map is saved as a NumPy array for direct use in Python.



#### Examples

##### Example 1: Basic Usage with Default Settings

```bash
python extract_disparity.py --restore_ckpt checkpoints/sceneflow.pth
```

- Processes image pairs from `left/` and `right/` directories.
- Saves outputs in `output/`.
- Does not save NumPy arrays.

##### Example 2: Saving NumPy Arrays

```
python extract_disparity.py --restore_ckpt checkpoints/sceneflow.pth --save_numpy
```

- Saves disparity maps as both PNG images and NumPy arrays.

##### Example 3: Custom Image Directories and Output Directory

```
python extract_disparity.py \
  --left_imgs /path/to/your/left_images/ \
  --right_imgs /path/to/your/right_images/ \
  --output_directory /path/to/save/outputs/ \
  --restore_ckpt checkpoints/sceneflow.pth
```

- Processes images from specified directories.
- Saves outputs in the specified output directory.

------

### to_pointcloud.py

#### Command-Line Arguments

The script accepts several command-line arguments to customize its behavior:

- `--in`, `--input_folder`: Input folder containing disparity maps (default: `output`).
- `--out`, `--output_folder`: Output folder to save point clouds (default: `output_pointcloud`).
- `--fx`: Focal length in the x-direction (default: `425.99684953503163`).
- `--fy`: Focal length in the y-direction (default: `426.0108446650122`).
- `--cx`: Optical center x-coordinate (default: `426.5960073761994`).
- `--cy`: Optical center y-coordinate (default: `240.4590369784203`).
- `--baseline`: Baseline between the stereo cameras in meters (default: `0.05000244116935238`).
- `--extrinsic_yaml_file`: Path to the YAML file containing the extrinsic matrices.
- `--camera_id`: Camera ID to use from the YAML file (default: `body_T_cam0`).
- `--denoise`: A flag to enable denoising.
- `--nb_neighbors`: Number of neighbors to analyze for each point.
- `--std_ratio`: The threshold based on the standard deviation of distances.

**Note**: All parameters have default values. You can override them by specifying the corresponding arguments.



#### Usage Examples

##### Basic Usage

To process disparity maps using default parameters:

```bash
python to_pointcloud.py
```

This command:

- Processes all `.dspm` files in the `output` directory.
- Saves the generated point clouds to the `output_pointcloud` directory.
- Uses the default intrinsic and extrinsic parameters specified in the script.

##### Specifying Intrinsic Parameters

If you want to specify custom intrinsic parameters:

```bash
python to_pointcloud.py --fx 500 --fy 500 --cx 320 --cy 240 --baseline 0.05
```

This command sets:

- Focal lengths `fx` and `fy` to `500`.
- Optical centers `cx` and `cy` to `320` and `240`, respectively.
- Baseline to `0.05` meters.

##### Using Extrinsic Matrices from a YAML File

To use extrinsic matrices defined in a YAML configuration file:

1. **Prepare your YAML file** (e.g., `config.yaml`) with the extrinsic matrices in the OpenCV matrix format:

   ```yaml
   body_T_cam0: !!opencv-matrix
     rows: 4
     cols: 4
     dt: d
     data: [0.00137718, -0.02366516, 0.99971899, 0.13932612,
            -0.99998467, 0.00532906, 0.00150369, 0.015785,
            -0.00536315, -0.99970574, -0.02365746, 0.00489279,
            0., 0., 0., 1.]
   ```

2. ##### **Run the script with the YAML file and specify the camera ID**:

   ```bash
   python to_pointcloud.py --extrinsic_yaml_file config.yaml --camera_id body_T_cam0
   ```

This command:

- Loads the extrinsic matrix `body_T_cam0` from `config.yaml`.
- Processes disparity maps using this extrinsic matrix.

##### Processing Multiple Files

The script automatically processes all `.dspm` files in the specified input directory.

- **Input Folder**: Use `--in` to specify the input directory.
- **Output Folder**: Use `--out` to specify the output directory.

**Example**:

```bash
python to_pointcloud.py --in disparity_maps --out pointclouds
```

This command:

- Processes all `.dspm` files in the `disparity_maps` directory.
- Saves the point clouds to the `pointclouds` directory.



#### Input and Output Formats

**Input: Disparity Map Files (`.dspm`)**

- The disparity map files should be in ASCII format.
- Each file contains rows of disparity values separated by spaces.
- Example content of a `.dspm` file:

  ```
  0.0 0.0 0.0 1.5 1.6 0.0
  0.0 2.1 2.2 0.0 0.0 0.0
  ```

#### Output: Point Cloud Files (`.ply`)

- The point clouds are saved in a custom ASCII PLY format.
- Each file starts with a PLY header followed by the point data.
- Example content of a `.ply` file:

  ```
  ply
  format ascii 1.0
  element vertex 12345
  property float x
  property float y
  property float z
  end_header
  x0 y0 z0
  x1 y1 z1
  ...
  ```



#### Notes

- **Dependencies**: Ensure all required Python packages are installed.
- **Extrinsic Matrix Format**: The YAML file must use the OpenCV matrix format for the extrinsic matrices.
- **Camera IDs**: When using a YAML file, ensure the `camera_id` you specify exists in the file.
- **Error Handling**: The script will raise errors if files are missing or formats are incorrect.
- **Extensibility**: You can extend the script to read intrinsic parameters from the YAML file by modifying the code (see comments in the script).



## Complete Example

Suppose you have the following setup:

- Disparity maps are stored in `./data/disparity_maps`.
- You want to save point clouds to `./data/pointclouds`.
- You have intrinsic parameters different from the defaults.
- You have a YAML file `camera_config.yaml` with extrinsic matrices.

Run the script as follows:

```bash
python to_pointcloud.py \
  --in ./data/disparity_maps \
  --out ./data/pointclouds \
  --fx 600.0 \
  --fy 600.0 \
  --cx 400.0 \
  --cy 300.0 \
  --baseline 0.06 \
  --extrinsic_yaml_file camera_config.yaml \
  --camera_id body_T_cam1
```

This command:

- Processes all `.dspm` files in `./data/disparity_maps`.
- Saves point clouds to `./data/pointclouds`.
- Uses the specified intrinsic parameters.
- Loads the extrinsic matrix `body_T_cam1` from `camera_config.yaml`.



## Additional Notes

- **Image Formats**: The script supports image files that can be opened by the Pillow library (e.g., `.png`, `.jpg`).
- **Grayscale Images**: If the images are grayscale, the script converts them to RGB by replicating the single channel.
- **Image Size**: Images are padded internally to be divisible by 32 for model compatibility.
