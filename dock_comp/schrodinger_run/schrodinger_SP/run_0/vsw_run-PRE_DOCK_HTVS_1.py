
import os
import pickle
import sys

script = sys.argv[0]

try:
    import schrodinger.pipeline.stages.gencodes
except ImportError:
    raise ImportError(script + ': Could not import module <schrodinger.pipeline.stages.gencodes>.')

stagename = os.path.splitext(script)[0]
restart_file = stagename + '.dump'

try: # Load the stage dump file:
    with open(restart_file, "rb") as fh:
        stage = pickle.load(fh)
except Exception:
    raise RuntimeError(script + ': Could not load stage from dump file')

######### MODIFY THIS SO THAT THE OPTIONS ARE UPGRADED EVEN WHEN RESTARTING ###

if not stage.hasStarted(): # If NOT restarting
    print('Stage', stage.stagename, 'initializing...')

    for position, obj in stage.iterInputs():
        obj.check() # Check to see if the object is valid

else: # Restarting
    print('Stage', stage.stagename, 'preparing to restart...')

# Periodically dump this instance to the dump file:

# Run the instance:
try:
    outputs = stage.run(restart_file=restart_file)
except RuntimeError as err:
    print(err) # Print the error without traceback
    sys.exit(1) # Exit this script

# Dump the outputs to a dump file:
try:
    with open(stagename + '.out', 'wb') as fh:
        pickle.dump(outputs, fh, protocol=2)
except Exception:
    raise RuntimeError(script + ': Could not write the output file')
