# Code for Experiment 3 (random labeling case) 

To generate results of our paper proceed with the following

- Run main_exp.py file and pass the appropriate config file (we provide two files `config_good_minst.yaml` for non random label MNIST dataset, and `config_random_mnist.yaml` for random label MNIST dataset).
```
python main_exp.py --config config_random_mnist.yaml
``` 
The reuslts of experiments will be saved in the `results` folder.
By default the experiment run on 'cuda:0' if GPU is availabel, you can change it in `main_exp.py` file.

You can change the parameters in config files to run experiments on some different parameters.

- Now open `infer.ipynb` and run all cells to generate plots
The file has code to generate plots for both experiments, you just need to give the correct config, by default it takes `config_random_mnist.yaml` for random label MNIST dataset and `config_good_mnist.yaml` for non-random label MNIST dataset. (Read the description in file for more details)
