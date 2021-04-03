import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
def plotUpvotes(df, bins):
    bins, counts = df.select('upvotes').rdd.flatMap(lambda x: x).histogram(bins)
    fig, ax = plt.subplots()

    freq, bins, patches = plt.hist(bins[:-1], edgecolor='white', label='1231231', bins=bins, weights=counts)
    plt.xlabel('Upvotes')
    plt.ylabel('Count')
    plt.title('Upvote Distribution')
    counts = np.array(counts)
    # Set the ticks to be at the edges of the bins.
    ax.set_xticks(bins)
    # Set the xaxis's tick labels to be formatted with 1 decimal place...
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    # Label the raw counts and the percentages below the x-axis...
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        # Label the raw counts
        ax.annotate(str(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -18), textcoords='offset points', va='top', ha='center')

        # Label the percentages
        percent = '%0.0f%%' % (100 * float(count) / counts.sum())
        ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -32), textcoords='offset points', va='top', ha='center')


    # Give ourselves some more room at the bottom of the plot
    plt.subplots_adjust(bottom=0.15)
    plt.show()
