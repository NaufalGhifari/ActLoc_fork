<p align="center">
  🚧 <b>Work in Progress</b> 🚧
</p>
<p align="center">
  <h2 align="center">ActLoc: Learning to Localize on the Move via Active Viewpoint Selection</h2>
  <h5 align="center">Conference on Robot Learning (CoRL) 2025</h5>
</p>

<div align="center"> 

<a href="https://boysun045.github.io/ActLoc-Project/"><b>Project Page</b></a> | 
<a href="https://www.arxiv.org/abs/2508.20981"><b>Paper</b></a> | 
<a href="#"><b>Demo (Coming Soon)</b></a>

</div>

<br>

<div align="center">

<p align="center">
  <a href="https://jiajieli7012.github.io/"><strong>Jiajie Li*</strong></a>
  ·
  <a href="https://boysun045.github.io/boysun-website/"><strong>Boyang Sun*</strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=cZeizisAAAAJ&hl=it"><strong>Luca Di Giammarino</strong></a>
  ·
  <a href="https://hermannblum.net/"><strong>Hermann Blum</strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=YYH0BjEAAAAJ"><strong>Marc Pollefeys</strong></a>
</p>
<p align="center"><strong>(* Equal Contribution)</strong></p>

</div>

---

## 🔄 Updates
- **Sep 15, 2025**: Initial code release

---

## 📝 TODO List
- [x] Single-viewpoint Selection Inference Code Release
- [x] Single-viewpoint Selection Demo Release
- [x] Path Planning Inference Code Release
- [x] Path Planning Demo Release
- [ ] Test Data and Evaluation Script Release
- [ ] Training Data and Training Script Release

---

## 🛠️ Setup
Our code has been tested on a workstation running **Ubuntu 22.04** with an **NVIDIA RTX 4090 (24GB)** GPU and **CUDA 12.4**.  

We recommend creating the environment from the provided `environment.yml` file:

```bash
# create environment
conda env create -f environment.yml

# activate environment
conda activate actloc_env
```

---

## Example Data Download
You can download the example data from [here](https://drive.google.com/drive/folders/1XhqJ-D92VykBgFU-moxfvp_SwavqTdl9?usp=sharing) and put it in the root folder of this repo to run the single-viewpoint selection inference code.

---

## Quick Start

### Step 1: Predict Poses from Waypoints

This script takes an SfM model and a list of waypoints as input, and predicts the optimal camera pose (extrinsic matrix) for each one.

```bash
python demo_single.py \
    --sfm-dir ./example_data/00005_reference_sfm \
    --waypoints-file ./example_data/waypoints.txt \
    --checkpoint ./checkpoints/trained_actloc.pth \
    --output-file ./example_data/selected_poses.npz
```

**Expected Output**: This will create a file named `selected_poses.npz` in the `example_data` directory, which contains the calculated poses.

### Step 2: Render Images from Predicted Poses

Now, use the `selected_poses.npz` file generated in the previous step to render images from the 3D mesh.

```bash
python capture_predicted_views.py \
    --mesh-file ./example_data/yPKGKBCyYx8.glb \
    --poses-file ./example_data/selected_poses.npz \
    --output-folder ./example_data/best_viewpoint_images
```

**Expected Output**: This will create a new folder named `best_viewpoint_images` containing the rendered images for each successful waypoint.

### Predict for Multiple Poses with Motion Constraints
```bash
python demo_multi.py \
    --sfm-dir ./example_data/00005_reference_sfm \
    --waypoints-file ./example_data/waypoints.txt \
    --checkpoint ./checkpoints/trained_actloc.pth \
    --output-file ./example_data/selected_poses.npz \
    --lamda 0.03
```

### Visualization
```bash
python vis.py \
    --meshfile ./example_data/yPKGKBCyYx8.glb \
    --waypoints-file ./example_data/waypoints.txt \
    --poses-file ./example_data/selected_poses.npz 
```
---

## Citation
If you find our paper useful, please consider citing:

```bibtex
@misc{li2025actloclearninglocalizeactive,
      title={ActLoc: Learning to Localize on the Move via Active Viewpoint Selection}, 
      author={Jiajie Li and Boyang Sun and Luca Di Giammarino and Hermann Blum and Marc Pollefeys},
      year={2025},
      eprint={2508.20981},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2508.20981}, 
}
```

