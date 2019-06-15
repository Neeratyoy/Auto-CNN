# Auto-configuration of CNNs [Automated Algorithm Design] 

The following repo is a sanitized version of the project submitted for the AutoML coursework at University of Freiburg.
The src/ folder contains the relevant source code and further details of the code structure.

As on overview:

* The task was to train CNN models for two datasets [KMNIST] and [K49]
* With minimal or no manual tuning for the architecture or parameters
* The problem was approached as a Hyper-parameter Optimization (HPO) task 
    * With Neural Architecture Search (NAS) as HPO
    * Tuning of training parameters as HPO 
* [BOHB] was used as a tool for HPO
* Overall, transfer learning was leveraged to optimize performance for the datasets
    

_Hardware used_:

* All scripts were confined to run one CPU at a time
* Utilizing a single core of Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz


_Compute budget_:

* A maximum budget of 24 hours on the above specification


_Results obtained_:

Dataset | Keras simple CNN [benchmark] | Auto-CNN |
--- | :---: | :---: | 
KMNIST | 95.12% | 97.89% |
K49 | 89.25% | 94.28% |  

_Note_: Results are using simple NAS without skip connections, or specialized architectures (using which can improve results and may increase compute time too)


For KMNIST:
A simple BOHB was run for 20 iterations.


For K49:
The best returned configuration from KMNIST was chosen. The size of the channels, number of fully connected layers and neurons, batch size was reparameterized and input to BOHB for K49. 



[KMNIST]: https://github.com/rois-codh/kmnist
[K49]: https://github.com/rois-codh/kmnist
[BOHB]: https://automl.github.io/HpBandSter/build/html/optimizers/bohb.html
[benchmark]: https://github.com/rois-codh/kmnist#benchmarks--results-
