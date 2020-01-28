import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import gaussian_kde
from definitions import path_join


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), np.std(a,ddof=1)/np.sqrt(n)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def get_by_step(X, step=0, column=None):
    result = []
    for _x in X:
        if _x.shape[0] > step:
            if column:
                try:
                    result.append(_x[step, column])
                except:
                    print(step)
            else:
                result.append(_x[step])
    result = np.array(result)
    return result


def get_with_steps_size(x, min_steps, max_steps=500, column=None):
    result = []

    for i in range(x.shape[0]):
        res = np.array(x[i])
        if (res.shape[0] >= min_steps) and (res.shape[0] <= max_steps):
            if column:
                result.append(res[:, column])
            else:
                result.append(res)
    result = np.array(result)
    return result


def get_std_error(x):
    e = np.std(x)
    return e


def get_mae(output, Y, step=0, column=None):
    output = get_by_step(output, step, column=column)
    Y = get_by_step(Y, step, column=column)
    mae = np.sum(np.absolute(output - Y))/output.shape[0]
    return mae


def plot_graphs(graphs, labels, model_name, images_dir):
    for graph, label in zip(graphs, labels):
        line_style = '-'
        if label != 'true_y':
            line_style = '--'
        plt.plot(graph, linestyle=line_style, label=label)

    # min_ylim = min(y2) # for big values bug
    # max_ylim = max(y2)
    # plt.ylim(min_ylim, max_ylim)

    plt.grid()
    plt.legend()
    title = '{} modeling with different models'.format(model_name)
    plt.title(title)

    image_path = path_join(images_dir, title + '.png')
    plt.savefig(image_path)
    plt.gcf().clear()


def plot_distribution(x, title,images_dir):
    # density = gaussian_kde(x)
    plt.figure(figsize=(8, 8))
    sns.distplot(x, hist=False)
    plt.grid()
    plt.title(title)
    # plt.show()

    plt.savefig(images_dir + title + '.png')
    plt.gcf().clear()
    # density.covariance_factor = lambda: .25
    # density._compute_covariance()
    # plt.plot(x, density(x))
    # plt.show()


def biplot(x, y, title, images_dir):
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y)
    plt.grid()

    min_ylim = min(min(x), min(y))
    max_ylim = max(max(x), max(y))

    plt.ylim(min_ylim, max_ylim)

    plt.xlabel('true value')
    plt.ylabel('nn value')
    plt.title(title)

    # plt.show()
    image_path = path_join(images_dir, title + '.png')
    plt.savefig(image_path)
    plt.gcf().clear()


def plot_std_error(x, y, e, y_label, title):
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('Time (in hours)')

    ax = plt.subplot(111)
    ax.set_xlim(0, max(x)+12)

    ticks = [i*12 for i in range(len(x)+1)]
    plt.grid()
    plt.xticks(ticks)
    plt.errorbar(x, y, e, linestyle='-', linewidth='1', marker='o', markersize='3', capsize=3,
                 elinewidth=1,
                 markeredgewidth=3)
    plt.savefig('../Images/' + title + '.png')
    plt.gcf().clear()


def plot_std_error_v2(x, y, e, y_label, title):
    y1, y2 = y
    e1, e2 = e

    # red_patch = mpatches.Patch(color='red', label='SD')
    # blue_patch = mpatches.Patch(color='blue', label='RNN')
    # legend = plt.legend(handles=[red_patch, blue_patch])

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(111)
    ax.set_xlim(12, max(x)+12)
    # ymin = min(y2)
    # if min(y1) < ymin:
    #     ymin = min(y1)
    # ymax = max(y2)
    # if max(y1) > ymax:
    #     ymax = max(y1)
    ax.set_ylim(0, 200)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Time (in hours)')

    ticks = [i*12 for i in range(1, len(x)+1)]
    plt.grid()
    plt.xticks(ticks)
    plt.errorbar(x, y1, e1, color='blue',linestyle='-', linewidth='1', marker='o', markersize='3', capsize=3,
                 elinewidth=1,
                 markeredgewidth=3, label='ANN Model')
    plt.errorbar(x, y2, e2, color='red', linestyle='-', linewidth='1', marker='o', markersize='3', capsize=3,
                 elinewidth=1,
                 markeredgewidth=3, label='SD Model')
    ax.legend(loc='upper right')

    plt.savefig('../simulations/step_by_step/images/' + title + '.png')
    plt.gcf().clear()


def plot_std_error_v3(x, y, e, y_label, title):
    e = np.array(e)
    y = np.array(y)
    e1, e2 = e
    y1, y2 = y
    # spread = x
    center = y1
    flier_high = y1+e1
    flier_low = y1-e1
    data = np.concatenate((center, flier_high, flier_low), 0)

    data1 = np.array([center, flier_high, flier_low])

    center = y2
    flier_high = y2+e2
    flier_low = y2-e2
    data2 = np.array([center, flier_high, flier_low])

    fig = plt.figure(figsize=(8, 8))


    ax = fig.add_subplot(111)
    ax.set_xlim(12, max(x) + 12)

    ax.boxplot(data)
    ax.set_title(y_label)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                        hspace=0.04, wspace=0.03)

    plt.savefig('../simulations/step_by_step/images/' + title + '.png')
    plt.gcf().clear()


def setBoxColors(bp):
    from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    setp(bp['fliers'][1], color='red')
    setp(bp['medians'][1], color='red')


def plot_std_error_v4(x, y, e, y_label, title):
    from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

    # function for setting the colors of the box plots pairs
    e = np.array(e)
    y = np.array(y)
    e1, e2 = e
    y1, y2 = y
    # spread = x
    center1 = y1
    flier_high1 = y1 + e1
    flier_low1 = y1 - e1
    data1 = np.array([center1, flier_high1, flier_low1])

    center2 = y2
    flier_high2 = y2 + e2
    flier_low2 = y2 - e2
    data2 = np.array([center2, flier_high2, flier_low2])
    # Some fake data to plot
    figure(figsize=(8, 8))
    ax = axes()
    hold(True)

    # bp = boxplot([data1[:, 0], data2[:, 0]], positions=[0.7, 1.1], widths=0.2)
    # bp = boxplot([data1[:, 1], data2[:, 1]], positions=[1.6, 2.0], widths=0.2)
    # bp = boxplot([data1[:, 2], data2[:, 2]], positions=[2.5, 2.9], widths=0.2)
    # bp = boxplot([data1[:, 3], data2[:, 3]], positions=[3.4, 3.8], widths=0.2)
    # bp = boxplot([data1[:, 4], data2[:, 4]], positions=[4.3, 4.7], widths=0.2)
    # bp = boxplot([data1[:, 3], data2[:, 3]], positions=[3.4, 3.8], widths=0.2)

    ax.set_title(y_label)

    ticks = [str(i * 12) for i in range(1, len(x) + 2)]
    _xs = [i for i in range(1, len(x)+2)]


    # bp = []
    widths = 0.2
    for i in range(1, len(_xs)):
         now = ticks[i-1]
         bp = boxplot([data1[:, i-1], data2[:, i-1]], positions = [i+0.2*widths, i+1.6*widths], widths = widths)
         setBoxColors(bp)
    # # draw temporary red and blue lines and use them to create a legend
    # hB, = plot([1,1],'b-')
    # hR, = plot([1,1],'r-')
    # legend((hB, hR),('Apples', 'Oranges'))
    # hB.set_visible(False)
    # hR.set_visible(False)
    # fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
    #                     hspace=0.04, wspace=0.03)



    xlim(1, 10)
    # ylim(-0.4, 1.5)
    ax.set_xticklabels(ticks)
    ax.set_xticks(_xs)
    ax.set_title(title)
    # draw temporary red and blue lines and use them to create a legend
    hB, = plot([1, 1], 'b-')
    hR, = plot([1, 1], 'r-')
    legend((hB, hR), ('ANN Model', 'SD Model'))
    hB.set_visible(False)
    hR.set_visible(False)

    plt.grid()
    savefig('results/simulations/step_by_step/images/' + title + '.png')
    # show()

def demo(X, steps, y_label, title, column=0):
    xs = np.arange(12, (steps+1)*12, 12)
    ys = X[:steps, column]

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot(xs, ys)
    ax.xticks(xs)
    ax.title(title)
    ax.ylabel(y_label)
    ax.xlabel('Time (in hours)')
    ax.savefig('../simulations/step_by_step/images/' + title + '.png')
    ax.gcf().clear()

def demo2(params, steps, y_label, title, images_dir, column=0):
    xs = np.arange(12, (steps+1)*12, 12)

    rnns = params[0]
    # sds = params[1]
    ys = params[1]

    rnns = rnns[:steps, column]
    # sds = sds[:steps, column]
    ys = ys[:steps, column]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    ticks = [i * 12 for i in range(1, len(xs) + 1)]
    ax.set_xticks(ticks)
    ax.set_xlim(12, max(xs) + 12)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Time (in hours)')

    ax.plot(xs, rnns, color='blue', label='ANN Model')
    # ax.plot(xs, sds, color='red', label='SD Model')
    ax.plot(xs, ys, color='green', label='Correct value')
    ax.grid(True)

    # ticklines = ax.get_xticklines() + ax.get_yticklines()
    # gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    # ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

    ax.legend(loc='upper right')

    plt.savefig(images_dir + title + '.png')
    plt.gcf().clear()

def get_statictics(output, Y, min_steps, max_steps, steps, column=None):

    ys = []
    es = []

    _output = get_with_steps_size(output, min_steps, max_steps)
    _Y = get_with_steps_size(Y, min_steps, max_steps)

    for i in steps:
        y = get_by_step(_output, step=i, column=column)
        _y = get_by_step(_Y, step=i, column=column)
        err = np.array((y-_y)**2)
        e_mean = err.mean()
        e = get_std_error(err)

        ys.append(e_mean)
        es.append(e)


    ms = []
    mes = []
    for i in steps:
        mae = get_mae(_output, _Y, step=i, column=column)
        # m, me = mean_confidence_interval(mae)
        ms.append(mae)
        mes.append(0)
    return ys, es, ms, mes

def again(x, title, name, images_dir):
    import numpy as np
    import pylab
    import scipy.stats as stats

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    stats.probplot(x, dist="norm", plot=pylab)

    pylab.ylim(0, max(x) + 1)
    pylab.ylabel(name)

    pylab.title(title)
    pylab.grid(True)
    pylab.savefig(images_dir + title + '.png')
    # pylab.show()