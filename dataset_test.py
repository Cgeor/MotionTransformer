from utils import two_ppl_3d as datasets
from utils.opt import Options
from utils.viz_2ppl import plot_predictions_from_3d
from matplotlib import pyplot as plt

def main(opt):
    dataset = datasets.Datasets(opt, split=0)
    data = dataset.p3d[1].numpy()

    fig = plt.figure()
    ax = plt.gca(projection='3d')
    plot_predictions_from_3d(data, data, fig, ax, '1')

if __name__ == '__main__':
    option = Options().parse()
    main(option)
