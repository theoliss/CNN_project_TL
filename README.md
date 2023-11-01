# CNN_Project_Letter_Recognition : 

### Installation :

- First be sure to use python3 and to have all the libraries listed in the requirements.txt, if not you can apply :

```bash
pip install torch
pip install onnx
pip install onnxruntime
pip install tqdm
```
-If you only want to execute the web page, torch and tqdm is not required
-It is highly recommanded to have cuda installed to increase greatly the execution time by using your GPU

### Execution : 

- ***To train to model :*** execute the "main.py" file using :
    
```bash
python3 main.py
```

- ***To execute the recognition in web :*** execute the "index.html" in a browser or use this link :

https://theoliss.github.io/CNN_project_TL/

### The program :

-The purpuse of this program is to generate and execute a simple Convolution Neural Network (CNN) based on the EMNIST dataset to recognize letters drawn by the user.
-The python file generate the neural network and save a "emnist.onnx" file containing the NN parameters
-The html and js scripts load this .onnx file in a web applicaton to have a exemple application of such a neural network