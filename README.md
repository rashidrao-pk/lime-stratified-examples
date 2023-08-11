# lime-stratified-examples
Examples used to generate plots for paper 'Using stratified sampling to improve LIME Image Explanations '

# Required Libraries
Install all the required libraries by running following line:
```
pip install -r requiremnets.txt 
```
# Installation

Download zipped file from XXXXXX:
```
wget www.xyz.com/lime-startified-with-examples
unzip -tar XXXXXXXX
```
in lime-startified/lime/lime-image.py search (CTRL+F) for following line:
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
# Run Examples
1. Run the Examples.ipynb notebook to run the extensive experiments on all seelcted 150 images.
2. Run Paper_Figures to create figures presented in the paper
