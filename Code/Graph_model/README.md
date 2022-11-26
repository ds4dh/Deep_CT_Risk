## ML

The machine learning part, including the data-related and network-related stuff.

### data:
- `ML.data.features` contains two classes to featurize free text to vectorial representations.
- `ML.data.flat` contains torch-style datasets for flat (non-graph) data.
- `ML.data.graph` contains the torch and torch geometric style dataset classes.

### networks:
- `ML.networks.flat` contains networks for flat-style text processing. These are essentially Multi-Layer Perceptrons 
with proper weight-sharing between the fields.
- `ML.networks.graph` contains graph NN's with Message Passing mechanisms to go from the graph-level to the final graph
or node-level representations.
