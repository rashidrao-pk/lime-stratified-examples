<center>
  <img src ='https://github.com/rashidrao-pk/lime-stratified-examples/assets/25412736/001bfdc7-5e73-4eab-957c-7efc7fed6673' width=210, height=100>
</center>

<h1> lime-stratified-examples</h1>
Examples used to generate plots for paper 'Using stratified sampling to improve LIME Image Explanations '

<h2> 🔧Dependencies and Installation </h2>
Python<br>
Tensorflow<br>
Option: NVIDIA GPU + CUDA <br>
Option: Windows <br>
<br>
Install all the required libraries by running following line:

```
pip install -r requirements.txt 
```

<h3> Installation </h3>
1. Clone repo<br>

Download zipped file from XXXXXX:
```
wget www.xyz.com/lime-startified-with-examples
unzip -tar XXXXXXXX
```
in lime-startified/lime/lime-image.py search for following line:
```
return ret_exp
```
and replace with following line:
```
return (data,labels,ret_exp)
```
save the file and then build and install it using following lines:
```
python setup.py build
python setup.py install
```
<h2> Run Examples </h2>

```
1. Run the Examples.ipynb notebook to run the extensive experiments on all seelcted 150 images
2. Run Paper_Figures to create figures presented in the paper
```

<h2>Data Availability:</h2>
<a href = '#'> 📝 Paper </a>: <br>
<a href = '#'>⚡ Codes </a> <br>
<a href = '#'> 📘Example Codes </a><br>

<h2>Authors:</h2>

|  Author  | Affiliation | Profile |
|  :-------- | :------:  | :--: |
| --------- |  --------- | --------- |


<h2>📜 License and Acknowledgement</h2>
<p>LIME-Stratified is released under MIT License 2024</p>

<h3>BibTeX:</h3>

```
xxx
```
