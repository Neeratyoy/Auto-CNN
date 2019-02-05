import numpy as np

from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from main import train
from main import eval
import torch

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()

# We define a few possible types of SVM-kernels and add them as "kernel" to our cs
lr = UniformFloatHyperparameter("learning_rate", 0.0001, 0.1, default_value=0.001)
cs.add_hyperparameter(lr)


def train_from_cfg(config):
    tr_acc, tr_loss, val_acc, val_loss, tr_time, val_time, params = train(
        dataset='KMNIST',  # dataset to use
        model_config={'n_layers': 2, 'n_conv_layer': 1},
        data_dir='../data',
        num_epochs=20,
        batch_size=50,
        learning_rate=config['learning_rate'],
        # train_criterion=torch.nn.MSELoss,
        train_criterion=torch.nn.CrossEntropyLoss,
        model_optimizer=torch.optim.Adam,
        data_augmentations=None,  # Not set in this example
        save_model_str=None
    )
    return(float(val_loss))

# Scenario object
# scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
#                      "runcount-limit": 20,  # maximum function evaluations
#                      "cs": cs,               # configuration space
#                      "deterministic": "true"
#                      })
#
# print("Optimizing! Depending on your machine, this might take a few minutes.")
# smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
#         tae_runner=train_from_cfg)



##############################################################################################

# # Build Configuration Space which defines all parameters and their ranges
# cs = ConfigurationSpace()
#
# # We define a few possible types of SVM-kernels and add them as "kernel" to our cs
# kernel = CategoricalHyperparameter("kernel", ["linear", "rbf", "poly", "sigmoid"], default_value="poly")
# cs.add_hyperparameter(kernel)
#
# # There are some hyperparameters shared by all kernels
# C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
# shrinking = CategoricalHyperparameter("shrinking", ["true", "false"], default_value="true")
# cs.add_hyperparameters([C, shrinking])
#
# # Others are kernel-specific, so we can add conditions to limit the searchspace
# degree = UniformIntegerHyperparameter("degree", 1, 5, default_value=3)     # Only used by kernel poly
# coef0 = UniformFloatHyperparameter("coef0", 0.0, 10.0, default_value=0.0)  # poly, sigmoid
# cs.add_hyperparameters([degree, coef0])
# use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
# use_coef0 = InCondition(child=coef0, parent=kernel, values=["poly", "sigmoid"])
# cs.add_conditions([use_degree, use_coef0])
#
# # This also works for parameters that are a mix of categorical and values from a range of numbers
# # For example, gamma can be either "auto" or a fixed float
# gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default_value="auto")  # only rbf, poly, sigmoid
# gamma_value = UniformFloatHyperparameter("gamma_value", 0.0001, 8, default_value=1)
# cs.add_hyperparameters([gamma, gamma_value])
# # We only activate gamma_value if gamma is set to "value"
# cs.add_condition(InCondition(child=gamma_value, parent=gamma, values=["value"]))
# # And again we can restrict the use of gamma in general to the choice of the kernel
# cs.add_condition(InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"]))
#
#
# def svm_from_cfg(cfg):
#     """ Creates a SVM based on a configuration and evaluates it on the
#     iris-dataset using cross-validation.
#
#     Parameters:
#     -----------
#     cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
#         Configuration containing the parameters.
#         Configurations are indexable!
#
#     Returns:
#     --------
#     A crossvalidated mean score for the svm on the loaded data-set.
#     """
#     # For deactivated parameters, the configuration stores None-values.
#     # This is not accepted by the SVM, so we remove them.
#     cfg = {k: cfg[k] for k in cfg if cfg[k]}
#     # We translate boolean values:
#     cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
#     # And for gamma, we set it to a fixed value or to "auto" (if used)
#     if "gamma" in cfg:
#         cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
#         cfg.pop("gamma_value", None)  # Remove "gamma_value"
#
#     clf = svm.SVC(**cfg, random_state=42)
#     d = datasets.load_iris()
#     iris_data = d['data']
#     iris_target = d['target']
#     scores = cross_val_score(clf, iris_data, iris_target, cv=5)
#     return 1 - np.mean(scores)  # Minimize!

# Scenario object
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 200,  # maximum function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "true"
                     })

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
        tae_runner=train_from_cfg)
        # tae_runner=svm_from_cfg)

incumbent = smac.optimize()

inc_value = svm_from_cfg(incumbent)

print("Optimized Value: %.2f" % (inc_value))

# We can also validate our results (though this makes a lot more sense with instances)
smac.validate(config_mode='inc', validate_insances = 'train+test')
