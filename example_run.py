import snet as sn
import connectome as cn
from parameters import load_default_params
import pickle

# Set some parameters
savepath = '.' # directory where you want to save the data
fdrive = 2.0
sdrive = 0.5

# Load in the default parameters
# Available models are "1pop", "2pop", and "8pop"
params = load_default_params("2pop")

# Build the Spine model
Spine = sn.SpinalNetwork(params)

# Change parameters
Spine.params['fastdrivemod'] = fdrive
Spine.params['slowdrivemod'] = sdrive
Spine.params['modelname'] = Spine.params['modelname'] + f"_fd{fdrive}_sd{sdrive}"

# Run the simulation
Spine.update_model() # Necessary to ensure that all parameter changes are properly applied
Spine.simulation()

# Save the output
with open(f'{savepath}/{Spine.params["modelname"]}.pkl', 'wb') as output:
    pickle.dump(Spine, output, protocol=4)
    print(f'spine pickled as {Spine.params["modelname"]}')
