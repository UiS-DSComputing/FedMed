from typing import Callable, Dict, List, Optional, Tuple

import os
import numpy as np
import flwr as fl
from flwr.common import Metrics
from flwr.server.utils import tensorboard
from flwr.server.strategy import FedAvg
from collections import OrderedDict
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple, Union
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"./global_model/round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics



class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(DEVICE)

def load_parameters_from_disk():
    # import Net
    list_of_files = [fname for fname in glob.glob('./round-*')]
    latest_round_file = max(list_of_files, key=os.path.getctime)
    parameters = np.load(latest_round_file)
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(parameters[v]) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net


# Create strategy and run server
if not os.path.exists("./global_model"):
    os.mkdir("./global_model")
LOGDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "flwr_logs")
strategy = tensorboard(logdir=LOGDIR)(SaveModelStrategy)()



# Start Flower server
if __name__ == "__main__":
    fl.server.start_server(
        server_address="[::]:8085",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )


