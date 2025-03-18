# FIFA Skeletal Tracking Starter Kit

This repository provides a **naïve baseline** for the **FIFA Skeletal Tracking Challenge**. It includes a simple, fully documented implementation to help participants get started with 3D pose estimation using bounding boxes, skeletal data, and camera parameters.

## 📌 Features
- **Baseline Implementation**: A simple approach for 3D skeletal tracking.
- **Camera Pose Estimation**: Computes camera transformations from bounding box correspondences.
- **Lane Line Refinement**: Improves camera rotation using detected lane lines.
- **Pose Projection & Optimization**: Projects 3D skeletons onto 2D images and refines translation via optimization.

## 🚀 Getting Started

### 📦 Installation
Make sure you have the required dependencies installed:

```bash
pip install numpy torch opencv-python tqdm scipy
```

## 📂 Data Preparation
The script expects the following dataset structure:

```
data/
│── cameras/
│   ├── sequence1.npz
│   ├── sequence2.npz
│── images/
│   ├── sequence1/
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│── skel_2d.npz
│── skel_3d.npz
│── boxes.npz
│── pitch_points.txt
```

- **`images/`**: Stores frame images for each sequence. **Please Ensure that the filenames are sequentially numbered** (e.g., `"00000.jpg"`, `"00001.jpg"`, etc.).
- **`cameras/`**: Contains `.npz` files with camera parameters for each sequence.
- **`boxes.npz`**: Stores bounding box data for each sequence.
- **`skel_2d.npz`**: Contains estimated 2D skeletal keypoints. 
- **`skel_3d.npz`**: Contains estimated 3D skeletal keypoints. 

You can find details about the `cameras`, `bounding boxes`, and `images` on the **Kaggle** page. For `skel_2d.npz` and `skel_3d.npz`, you can generate them automatically using the provided `preprocess.py` script. Alternatively, we have also uploaded preprocessed data [here](google-drive-link).

### 📺 Sample Visualization
To help you visualize the results, we provide a short sample sequence in `media/sample.mp4`. 

## 🔧 Running the Baseline
To run the baseline model on the dataset, simply execute:

```bash
python baseline.py
```

By default, the script reads from the data/ directory and generates a `.npz` file (`dummy-solution.npz`) in the root folder:

You can then use the `prepare-submission.py` to create a submission file:

```bash
python prepare-submission.py -i dummy-solution.npz
```

## 📌 Notes
- This is a **naïve baseline** — you are encouraged to improve the accuracy by refining camera estimation, leveraging better keypoint tracking, or integrating deep learning approaches.

## 🤝 Contributing
If you find a bug or have suggestions for improvements, feel free to submit a pull request or open an issue.

## Acknowledgement
We use [4DHuman](https://github.com/shubham-goel/4D-Humans/tree/main) in the `preprocess.py` for estimating both 2D and 3D skeletons from bounding boxes. We appreciate the contributions of the developers and the broader research community in advancing human pose estimation.

## 📜 License
This project is licensed under the MIT License.
