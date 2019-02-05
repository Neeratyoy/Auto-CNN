import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis


def generateViz(result, out_dir = '.'):

    # load the example run from the log files
    # result = hpres.logged_results_to_HBS_result(out_dir)

    # get all executed runs
    all_runs = result.get_all_runs()

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()


    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()

    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]


    # We have access to all information: the config, the loss observed during
    #optimization, and all the additional information
    inc_config = id2conf[inc_id]['config']
    # inc_loss = inc_run.info['valid_loss']
    # inc_loss = inc_run.info['validation_loss']
    # inc_test_loss = inc_run.info['test_loss']
    # inc_accuracy = inc_run.info['valid_accuracy']
    # inc_test_accuracy = inc_run.info['test_accuracy']
    #
    # print('Best found configuration:')
    # print(inc_config)
    # print('Best loss value: %f (validation) and %f (test).'%(inc_loss, inc_test_loss))
    # print('It achieved accuracies of %f (validation) and %f (test).'%(inc_accuracy, inc_test_accuracy))

    # Let's plot the observed losses grouped by budget,
    hpvis.losses_over_time(all_runs)
    plt.tight_layout()
    plt.savefig(out_dir+'/plot_losses_over_time.png', dpi=150)

    # the number of concurent runs,
    hpvis.concurrent_runs_over_time(all_runs)
    plt.tight_layout()
    plt.savefig(out_dir + '/plot_concurrent_runs_over_time.png', dpi=150)

    # and the number of finished runs.
    hpvis.finished_runs_over_time(all_runs)
    plt.tight_layout()
    plt.savefig(out_dir + '/plot_finished_runs_over_time.png', dpi=150)

    # This one visualizes the spearman rank correlation coefficients of the losses
    # between different budgets.
    hpvis.correlation_across_budgets(result)
    figure = plt.gcf()
    figure.set_size_inches(10, 10)
    plt.savefig(out_dir + '/plot_correlation_across_budgets.png', dpi=150)

    # For model based optimizers, one might wonder how much the model actually helped.
    # The next plot compares the performance of configs picked by the model vs. random ones
    hpvis.performance_histogram_model_vs_random(all_runs, id2conf)
    figure = plt.gcf()
    figure.set_size_inches(10, 10)
    plt.savefig(out_dir + '/plot_performance_histogram.png', dpi=150)

    return
    # plt.show()
