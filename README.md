<center><img src ='lime logo 3.png' width=210, height=100></center>

# lime-stratified-examples
Code used to generate plots for paper 'Using stratified sampling to improve LIME Image Explanations'.

### 🔧 Dependencies and Installation
- Python
- Tensorflow
- Option: NVIDIA GPU + CUDA

Install all the required libraries by running following line:

```
pip install -r requirements.txt 
```

### Structure of the artifact

This artifact is structured as follows:

- the `data/` folder contains a subset of the [ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) (the first 150 images) used to generate the plots and benchmarks of the paper;

- the `lime_stratified/` folder that contains a modified version of [lime](https://github.com/marcotcr/lime) with the proposed stratified sampling approach

- the `results/` folder, which contains the results after running the artifact.

- two notebooks:

  * `Run_Benchmark.ipynb` runs the (long) benchmark on a subset of the ImageNet Object Localization Challenge, generating the CSV file in the `results/` folder. 
  Inside the file there is an option to also run a "simplified" version of the benchmark (just one image) instead of the "full" version. The simplified version will run all the steps, but on a single image.

  * `Paper_Figures.ipynb` which reads the CSV file and generates the plots included in the paper.

The precomputed CSV file is already included in the `results/` folder.


---
### How to run the artifact

The `Run_Benchmark.ipynb` will take a long tame (more than 1 day) to run all the experiments and generate the CSV file. All explanations are computed 10 times and then averaged, to better stabilize the results and reduce randomization.

The `Paper_Figures.ipynb` has several targets: for the single explanations, it will take a few minutes. For the plots generated starting from the CSV file, in will take a few seconds.


---
### Run Examples

You can run `Run_Benchmark.ipynb` to regenerate all the results. Alternatively, the precomputed CSV file used for the paper is included.

Then run  `Paper_Figures.ipynb` to generate the plots.
