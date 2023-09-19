import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams.update({
    "text.usetex": True,  # requires "sudo apt install cm-super dvipng" on linux
    "text.latex.preamble": r"\usepackage[vvarbb]{newtx}",
    "font.family": "serif",
    "font.serif": ["Helvetica"],
    'figure.dpi': 200,  # renders images larger for notebook
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.titlesize': 10,
})

def prettify(ax):
    ax.spines['top'].set_alpha(.3)
    ax.spines['right'].set_alpha(.3)
    ax.spines['left'].set_alpha(.3)
    ax.spines['bottom'].set_alpha(.3)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', length=3)