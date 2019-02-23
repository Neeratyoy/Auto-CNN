import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
from matplotlib import gridspec
import numpy as np

def setBoxColors(bp, col1='blue', col2='red'):
    '''
    Function to annotate box plot components with different colours to demarcate train and validation
    :param bp: Boxplot object
    :param col1: Colour 1 for Training
    :param col2: Colour 2 for Testing
    :return: Nothing really - kind of pass by reference
    '''
    plt.setp(bp['boxes'][0], color=col1)
    plt.setp(bp['caps'][0], color=col1)
    plt.setp(bp['caps'][1], color=col1)
    plt.setp(bp['whiskers'][0], color=col1)
    plt.setp(bp['whiskers'][1], color=col1)
    plt.setp(bp['fliers'][0], markeredgecolor=col1)
    # plt.setp(bp['fliers'][1], color=col1)
    plt.setp(bp['medians'][0], color=col1)

    plt.setp(bp['boxes'][1], color=col2)
    plt.setp(bp['caps'][2], color=col2)
    plt.setp(bp['caps'][3], color=col2)
    plt.setp(bp['whiskers'][2], color=col2)
    plt.setp(bp['whiskers'][3], color=col2)
    plt.setp(bp['fliers'][1], markeredgecolor=col2)
    # plt.setp(bp['fliers'][3], color=col2)
    plt.setp(bp['medians'][1], color=col2)


def generateLossComparison(out_dir, show=False):
    '''
    Function to generate box plots over different budgets for an entire BOHB run
    :param out_dir: Directory where the plots are to be saved
    :param show: True/False to display the plots (additionally to saving)
    :return: void
    '''
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(out_dir)

    plot_data = {}
    for k in result.data.keys():
        try:
            sample = result.data[k].results
        except TryError:
            continue
        for b in sample:
            if sample[b] is None:
                continue
            if b not in plot_data.keys():
                plot_data[int(b)] = [[sample[b]['info']['train_loss']], [sample[b]['info']['test_loss']]]
            else:
                # print(k, b)
                plot_data[int(b)][0].append(sample[b]['info']['train_loss'])
                plot_data[int(b)][1].append(sample[b]['info']['test_loss'])

    max_loss = 0
    for i, k in enumerate(plot_data.keys()):
        max_loss = max(max_loss, np.max(np.array(plot_data[k])))

    fig = plt.figure(figsize=(10, 4), dpi=150)
    plt.suptitle("Loss comparison of Train and Validation over Epochs (Budget)")
    gs = gridspec.GridSpec(1, len(plot_data.keys()))
    for i, k in enumerate(plot_data.keys()):
        exec("ax"+str(i+1)+" = plt.subplot(gs[0, "+str(i)+"])")
        exec("ax"+str(i+1)+".grid(which='major', linestyle=':', axis='y')")
        exec("bp = ax"+str(i+1)+".boxplot(plot_data["+str(k)+"], showmeans=True, meanline=True, "+
             "sym='+', meanprops={'linestyle':'-'},whiskerprops={'linestyle': '--', 'color': 'blue'})")
        exec("ax"+str(i+1)+".set_ylim(0, "+str(max_loss+0.1)+")")
        exec("ax"+str(i+1)+".set_xlabel('Budget:"+str(k)+"')")
        if i==0:
            exec("ax"+str(i+1)+".set_ylabel('Loss')")
        exec("setBoxColors(bp)")
    hB, = plt.plot([1,1],'b-')
    hR, = plt.plot([1,1],'r-')
    hG, = plt.plot([1,1],'g-')
    plt.figlegend((hB, hR, hG),('Training', 'Validation', 'Mean'),loc='upper right')
    hB.set_visible(False)
    hR.set_visible(False)
    hG.set_visible(False)
    plt.savefig(out_dir+'/loss_comparison_plot.png',dpi=300)
    if show:
        plt.show()


def generateViz(out_dir, show=False):
    '''
    Generate plots for BOHB (from BOHB_visualizations.py from the documentations)
    :param out_dir: Directory to save the plots
    :param show: True/False to display the plots (additionally to saving)
    :return: void
    '''
    result = hpres.logged_results_to_HBS_result(out_dir)
    # get all executed runs
    all_runs = result.get_all_runs()
    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()
    # Let's plot the observed losses grouped by budget,
    hpvis.losses_over_time(all_runs)
    plt.tight_layout()
    plt.savefig(out_dir+'/plot_losses_over_time.png',dpi=300)
    # the number of concurent runs,
    hpvis.concurrent_runs_over_time(all_runs)
    plt.tight_layout()
    plt.savefig(out_dir + '/plot_concurrent_runs_over_time.png',dpi=300)
    # and the number of finished runs.
    hpvis.finished_runs_over_time(all_runs)
    plt.tight_layout()
    plt.savefig(out_dir + '/plot_finished_runs_over_time.png',dpi=300)
    # This one visualizes the spearman rank correlation coefficients of the losses
    # between different budgets.
    hpvis.correlation_across_budgets(result)
    figure = plt.gcf()
    figure.set_size_inches(10, 10)
    plt.savefig(out_dir + '/plot_correlation_across_budgets.png',dpi=300)
    # For model based optimizers, one might wonder how much the model actually helped.
    # The next plot compares the performance of configs picked by the model vs. random ones
    hpvis.performance_histogram_model_vs_random(all_runs, id2conf)
    figure = plt.gcf()
    figure.set_size_inches(10, 10)
    plt.savefig(out_dir + '/plot_performance_histogram.png', dpi=150)
    if show:
        plt.show()