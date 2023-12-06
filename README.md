# Residual Block based on Focus Attention Graph Convolution Neural Network(GCN-RBFA)

# Installation
This project required Pytorch

# Usage
For training and test, you need a dataset consist of structure files (POSCAR format, placed in ./data/DD-slab/) and related properties data (element_info_ND.xlsx、DFT_data.txt、Free_energy_Dataframe.csv, placed in ./data/make_row_data/). Then, you can enter the folder:./data/make_row_data/ and run training by following command:
```bash
python make_row_data.py
```

Then, you can get poscarlist.txt、DD_A.txt、DD_atten_scores.txt、DD_graph_indicator.txt、DD_graph_labels.txt、DD_node_labels.txt in the ./data/DD/, including graph name, edge index, node mask, graph index, icon label, and node label information.

You can train the model based on this information using the following command
```bash
python main.py
```

The trained model and prediction data are stored in ./result/ folder.

## Data Format

1. `element_info_ND.xlsx`:
This file consists of 7 columns, representing the element type, atomic radius, first ionization energy, electron affinity, electronegativity, number of valence electron layers, and the period in which the element is located respectively.

2. `DFT_data.txt`
You need to write a DFT_data.txt file in order to obtain the database. This file has 7 columns, which are metal composition, distance type, reaction site, DFT energy in the adsorption intermediate state, zero point energy, entropy contribution (TS), and slab DFT energy.

3. `Free_energy_Dataframe.csv`
You need to write Free_energy_Dataframe.csv file for the dataset. It is consist of nine columns. The first seven columns are feature values in dataset, and the last two columns are values of properties. Note that the number of columns can be adjusted according to the needs of your specific task.

