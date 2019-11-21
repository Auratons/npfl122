import subprocess
import argparse
import itertools
import importlib
import pandas as pd
import os
import sys
import plotly.io as pio
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


def parallel_coordinate_plot(gs_summary: pd.DataFrame) -> None:
    fig = make_subplots(rows=2, cols=1, specs=[
                        [{'type': 'domain'}], [{'type': 'domain'}]])
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(gs_summary.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=list(gs_summary.values.T),
                fill_color='lavender',
                align='left')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Parcoords(
            line=dict(
                color=gs_summary['score'],
                colorscale='Jet',
                showscale=True),
            dimensions=list(
                [
                    dict(
                        label=name,
                        values=gs_summary[name],
                        tickformat='d'
                    ) for name in gs_summary.columns
                ] +
                [
                    dict(
                        label='index',
                        values=gs_summary['index'],
                        tickvals=list(range(0, len(gs_summary)))
                    ),
                    dict(
                        label=f'score',
                        values=gs_summary['score']
                    )
                ]
            )
        ),
        row=2, col=1
    )

    filename = 'file.html'
    with open(filename, 'w') as f:
        f.write(fig.write_html(filename, include_plotlyjs=True,
                               full_html=True, auto_open=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--values_file", default='./labs/03/q_learning_gs_options.py',
                        type=str, help="A file with a dictionary of parameter name: list of values to test.")
    parser.add_argument("--script_name", default='./labs/03/q_learning.py',
                        type=str, help="Name of a script to test with all combinations.")
    args = parser.parse_args()

    # Perform import of an absolute path, expect only "data" variable defined inside.
    data = {}
    exec(open(os.path.abspath(args.values_file)).read())
    keys = list(data.keys())
    values = list(data.values())
    scores = []

    for combination in itertools.product(*values):
        command_list = [sys.executable, args.script_name] + \
            [f'--{keys[idx]}=' + str(val)
             for idx, val in enumerate(combination)]
        print(' '.join(command_list))
        result = subprocess.Popen(
            command_list,
            # We capture output inside 'result' variable, then save stdout and stderr inside files.
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        # Expected output is 'The mean 100-episode return after evaluation -227.96\n', other statistics
        # should be written to stderr.
        scores.append(result)

    scores = [float(result.stderr.split(' ')[-1]) for result in scores]

    gs_summary = pd.DataFrame(
        list(itertools.product(*values)),
        columns=list(data.keys())
    )
    gs_summary['score'] = scores
    gs_summary.sort_values(by=['score'], ascending=False)
    gs_summary.insert(0, 'index', pd.Series(list(range(0, len(gs_summary)))))

    parallel_coordinate_plot(gs_summary)
