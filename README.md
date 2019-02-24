---
output:
  pdf_document: default
  html_document: default
---
# ML4AAD WS18/19 Final Project

The following repo is a cloned extension of the original base repo that we were given. \newline
https://bitbucket.org/biedenka/ml4aad_ws18_project/src/master/

The \textit{src/} is maintained as per the original. However, the contents of the original \textit{src/} has been completely revamped. Details of which are given below.

The directory contains additional folders, namely:

* \textit{data/} where datasets.py dumped the KMNIST and K49 datasets
* \textit{plots/} contains certain images and plots that are being used for report generation
* \textit{reading/} certain reading material collected for my reference during literature survey
* \textit{report/} the .tex files and the final PDF of the presentation is contained here

The \textit{src/} folder contains the .py scripts that were created for this project and a folder called \textit{experiments/} which house the logged results of the experiments carried out. There is also a file called architecture.txt which contains the specfications of the hardware used to generate the numbers inside \textit{experiments/}. 



To run the default configuration given: 
```
python3 original_main.py python3 original_main.py -d K49
```


To run basic BOHB:

* The main interface is the bohb_main.py 
* It is dependent on main.py, BOHB_plotAnalysis.py, cnn.py 
* To run BOHB on K49 with eta=2, min_budget=3, max_budget=10, n_iterations=10, the following command can be given
```
python3 bohb_main.py --dataset K49 --eta 2 --min_budget 3 --max_budget 10 --n_iterations 10 --out_dir [path to write destination]
```


To run transfer learning:

* The main interface is the transfer_learning.py 
* It is dependent on main.py, main_transfer_learning.py, BOHB_plotAnalysis.py, cnn.py 
* To train BOHB's incumbent configuration on one dataset (source) and subsequently train on another dataset (dest) the following command can be given
```
python3 transfer_learning.py --dataset_source KMNIST --dataset_dest K49 --config_dir [path where BOHB results exist] --epochs_source 12 --epochs_dest 20
```


To run transfer of configuration:

* The main interface is the transfer_config.py
* It is dependent on main_transfer_config.py, BOHB_plotAnalysis.py, cnn_transfer_config.py
* To train BOHB's incumbent configuration with BOHB again with a reduced hyperparameter space, the following command can be given
```
python3 transfer_config.py --dataset K49 --eta 2 --min_budget 3 --max_budget 10 --n_iterations 10 --config_dir [path where BOHB results exist] --out_dir [path to write destination]
```


To evaluate any incumbent from BOHB by training on entire train set and testing on test:
```
python3 generate_result.py --dataset K49 --config_dir [path where BOHB results exist] --epochs 20
```
In case of testing a configuration from the run of transfer_config.py an additional argument '--transfer True' is needed
```
python3 generate_result.py --dataset K49 --epochs 20 --transfer True --config_dir [path where BOHB results exist] 
```



Other Python scripts are either called by the aforementioned primary scripts or have been used for other analysis (such as successive_halving.py)


The experiments/ folder contain logs and results of the analysis/experiments carried out for the project:

* This folder contains an excel sheet with the details or numbers for the experiments recorded
* This folder also contains two JSONs which contain the best performing configurations obtained for KMNIST and K49
* The folders are generally named in the format x_y_z where x, y, z are parameters to BOHB with x=eta, y=min_budget, z=max_budget


```
contact: neeratyoy@gmail.com
```