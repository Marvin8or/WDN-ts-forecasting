"""
The following example demonstrates how to import WNTR, create a water 
network model from an EPANET INP file, simulate hydraulics, and plot 
simulation results on the network.
"""
# Import WNTR
import wntr

# Create a water network model
inp_file = "examples/networks/Net1.inp"
wn = wntr.network.WaterNetworkModel(inp_file)
wn.options.time.duration = 2 * 24 * 3600

wn.options.time.hydraulic_timestep = 3600
wn.options.time.pattern_timestep = 3600
wn.options.time.report_timestep = 3600

# Simulate hydraulics
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()

# Disable controls
ctrls_to_disable = wn.control_name_list
# ctrl_1 = wn.get_control("control 1")
# ctrl_2 = wn.get_control("control 2")
for name in ctrls_to_disable:
    wn.remove_control(name)

# Plot results on the network
pressure_at_5hr = results.node["pressure"].loc[5 * 3600, :]
wntr.graphics.plot_network(
    wn,
    node_attribute=pressure_at_5hr,
    node_size=30,
    title="Pressure at 5 hours",
    node_labels=True,
    directed=True,
)


# %% Plot individual nodes

import matplotlib.pyplot as plt


def plot_node_timeseries(node_id: str, xlabel: str = "timestamps [s]") -> None:
    fig, axs = plt.subplots(4, 1, sharex=True)

    fig.suptitle(f"Node: {node_id}")
    for i, k in enumerate(results.node.keys()):
        axs[i].plot(
            results.node[k].index.values,
            results.node[k][node_id].values,
            "bo",
            label=k,
        )
        axs[i].set_ylabel(k)
        axs[i].grid()
        axs[i].legend()
    axs[-1].set_xlabel(xlabel)


node_id = "22"
plot_node_timeseries(node_id)

fig, axs = plt.subplots(1, 1, sharex=True)
tank_name = "2"
tank_elevation = wn.get_node(tank_name).elevation
tank_level = results.node["head"][tank_name] - tank_elevation
axs.plot(tank_level)
axs.axhline(33.528, color="black")
axs.axhline(42.672000000000004, color="black")


# %% List all nodes and print the pattern associated to it
import numpy as np

for node_name in wn.node_name_list:
    node_obj = wn.nodes.get(node_name)

    # Try to get the multipliers of a demand pattern attributed to a node
    try:
        base_demand = node_obj.demand_timeseries_list.base_demand_list()
        demand_pattern_name = node_obj.demand_pattern
        pattern = wn.get_pattern(demand_pattern_name)
        multipliers = pattern.multipliers
        print(node_name, demand_pattern_name, base_demand, len(multipliers))
    except AttributeError:
        continue

# %% visualization of node demands
from typing import Dict, List


# TODO generalize
def get_demand_pattern_matrix(
    wn: wntr.network.WaterNetworkModel, pattern: str, row_spacing: int = 5
):
    """Generate matrix used to visualize the demand patterns over the entire duration of the simulation"""

    # base demand * multipliers for each node
    demand_pattern = wntr.metrics.hydraulic.expected_demand(wn)

    nodes_with_selected_pattern = []

    for node_id in demand_pattern.columns:
        node = wn.get_node(node_id)

        if (
            hasattr(node, "demand_timeseries_list")
            and len(node.demand_timeseries_list) > 0
        ):
            pattern_name = node.demand_timeseries_list[0].pattern_name
            if pattern_name == pattern:
                nodes_with_selected_pattern.append(node_id)

    demand_pattern = demand_pattern[nodes_with_selected_pattern]

    # Create common time base
    duration = wn.options.time.duration
    hydraulic_timestep = wn.options.time.hydraulic_timestep
    common_time_base = np.arange(
        0, duration + hydraulic_timestep, hydraulic_timestep
    )

    num_rows = len(demand_pattern.columns) * row_spacing

    # Batches of rows to be filled with same values on axis=1, for better visibility
    row_batches = np.array(list(range(num_rows))).reshape(
        len(demand_pattern.columns), row_spacing
    )

    # Generate matrix and fill it with nan values
    demand_matrix = np.full(
        shape=(num_rows, len(common_time_base)), fill_value=np.nan
    )

    row_node_id_pair = []

    # Iterate batches, node names and fill matrix
    for row_batch, node_name in zip(row_batches, demand_pattern.columns):
        # Create temporary matrix
        tmp_mat = np.full(
            shape=(row_spacing, len(common_time_base)),
            fill_value=demand_pattern[node_name],
        )

        demand_matrix[row_batch, :] = tmp_mat
        row_node_id_pair.append([node_name, row_batch])

    return row_node_id_pair, demand_matrix


def visualize_demand_matrix(
    matrix: np.ndarray,
    node_names: List[List[any]],
    title: str,
    row_spacing: int = 5,
) -> None:
    fig, axs = plt.subplots(1, 1)
    fig.suptitle(title)
    im = axs.imshow(matrix)

    plt.colorbar(im)

    axs.set_yticks([node[1][0] for node in node_names])
    axs.set_yticklabels([node[0] for node in node_names])


pattern_names = list(wn.patterns.keys())  # ['1', '2', '3', '4', '5']

for pattern_name in pattern_names:
    row_node_id_pair, demand_matrix = get_demand_pattern_matrix(
        wn, pattern=pattern_name, row_spacing=1
    )
    # Visualize matrix
    visualize_demand_matrix(
        demand_matrix,
        row_node_id_pair,
        row_spacing=1,
        title=f"Pattern: {pattern_name}",
    )

# %% List all Valves
valve_names = wn.valve_name_list
# Print the valve names
print("List of valves in the network:")
for name in valve_names:
    print(name)

# Print control names
control_names = wn.control_name_list
for name, control in wn.controls():
    print(name, control)

ctrls_to_disable = ["control 1", "control 2"]
ctrl_1 = wn.get_control("control 1")
ctrl_2 = wn.get_control("control 2")
for name in ctrls_to_disable:
    wn.remove_control(name)

for name, control in wn.controls():
    print(name, control)

# Add controls
for name, ctrl in zip(ctrls_to_disable, [ctrl_1, ctrl_2]):
    wn.add_control(name, ctrl)

for name, control in wn.controls():
    print(name, control)
