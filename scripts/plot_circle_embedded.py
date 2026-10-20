import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import numpy as np
import rgg.algorithm as alg
import rgg.curves as curves
import rgg.sampling as samp
import rgg.plots as plots

if __name__=='__main__':
    print('generating embedded circle...')

    #parameters
    N=3000
    d=2000
    p=0.2
    k=2
    Tr_target=0.025
    circle_curve = curves.make_circle(scale=1)

    np.random.seed(1)
    X_circle, t_circle = samp.generate_manifold_samples(N, circle_curve)
    plots.plot_rgg_embedded(
        X_circle, 
        title="Circle", 
        d=d, 
        p=p, 
        k=k,
        color_by=t_circle, 
        Tr_target=Tr_target, 
        n_show=40, 
        seed=0,
        save_path="figures/circle_rgg_embedded_newtau_d2000.png"
    )

