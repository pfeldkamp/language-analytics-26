## !!! Run the following commands (one at a time) in the Terminal every time we start a session !!! 

### Change directory to your user folder
cd YourUserFolder#xxxx/ 

### Activate the virtual environment
source .venv/bin/activate

### Point Jupyter notebook files (.ipynb) to the correct .venv Python interpreter 
python -m ipykernel install --user --name=.venv

### Open a notebook file (.ipynb/.py) and make sure that .venv Python interpreter is a available.
When running .py files, chec the Python interpreter in the bottom right corner is from your .venv. If not, click and change it.



## for SESSION04, add:
sckit-learn

### if not already there, also these:
requests
nltk
ipywidgets 
spacy
datasets
seaborn

### install requirements if anything you added anything
pip install -r requirements.txt





# Deactivate environment if you need to switch environment
deactivate
