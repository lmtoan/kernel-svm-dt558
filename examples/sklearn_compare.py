"""
This module demonstrates the 3 types of KernelSVM: linear, polynomial order 3, and rbf,
by applying to a simulated 2D dataset.

And compare them to Scikit-Learn kernel SVM as indicated in the following link.

http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html.

All plots are stored in '../images' as default.

Implementation by Toan Luong
toanlm@uw.edu
June 2018
"""
from sim_demo import sim_demo

def sklearn(cache_plot_path, verbose=True, plot_contour=False):
    """
    Main function to run the real-data demonstration.

    Args:
        plot_cache_path: Path to store plots/images.
        verbose: Set to True for extensive outputs.
        plot_countour: Set to True for contour plots.
    """
    sim_demo(cache_plot_path, check_sklearn=True, verbose=verbose, plot_contour=plot_contour)

if __name__=='__main__':
    cache_path = '../images'
    sklearn(cache_path)