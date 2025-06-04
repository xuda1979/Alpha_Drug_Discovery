import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

from models import gan_drug_design, rl_drug_design
from alpha_drug_discovery import workflows
from data import dataset_download
import numpy as np

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Alpha Drug Discovery GUI"),
    html.Div([
        html.Label("Dataset"),
        dcc.Dropdown(
            id='dataset',
            options=[
                {'label': 'ESOL', 'value': 'esol'},
                {'label': 'BindingDB', 'value': 'bindingdb'}
            ],
            value='esol'
        ),
        html.Label("Epochs"),
        dcc.Input(id='epochs', type='number', value=1)
    ]),
    html.Button('Run GAN Design', id='run-gan'),
    html.Button('Run RL Design', id='run-rl'),
    html.Button('Run Pipeline', id='run-pipeline'),
    dcc.Textarea(id='output', style={'width': '100%', 'height': 200})
])


def _run_gan(dataset: str, epochs: int) -> str:
    df = dataset_download.load_dataset(dataset, use_sample=True)
    X = df.select_dtypes(float).values
    gan_drug_design.train_gan(X, epochs=epochs, batch_size=min(32, len(X)))
    return "GAN design complete"


def _run_rl(epochs: int) -> str:
    class DummyEnv:
        def __init__(self, state_dim: int, action_dim: int, max_steps: int = 5):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.max_steps = max_steps

        def reset(self):
            return np.zeros(self.state_dim)

        def step(self, action: int):
            next_state = np.random.rand(self.state_dim)
            reward = np.random.rand()
            done = False
            return next_state, reward, done, {}

    env = DummyEnv(10, 3)
    policy_network = rl_drug_design.PolicyNetwork(state_dim=10, action_dim=3)
    rl_drug_design.train_policy_gradient(env, policy_network, epochs=epochs)
    return "RL design complete"


def _run_pipeline() -> str:
    workflows.basic_drug_discovery_pipeline()
    return "Pipeline run complete"


@app.callback(Output('output', 'value'), Input('run-gan', 'n_clicks'),
              State('dataset', 'value'), State('epochs', 'value'))
def handle_gan(n_clicks, dataset, epochs):
    if not n_clicks:
        return ''
    return _run_gan(dataset, int(epochs))


@app.callback(Output('output', 'value', allow_duplicate=True), Input('run-rl', 'n_clicks'),
              State('epochs', 'value'), prevent_initial_call=True)
def handle_rl(n_clicks, epochs):
    if not n_clicks:
        return dash.no_update
    return _run_rl(int(epochs))


@app.callback(Output('output', 'value', allow_duplicate=True), Input('run-pipeline', 'n_clicks'),
              prevent_initial_call=True)
def handle_pipeline(n_clicks):
    if not n_clicks:
        return dash.no_update
    return _run_pipeline()


if __name__ == '__main__':
    app.run_server(debug=True)
