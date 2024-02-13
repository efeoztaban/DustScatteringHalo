import sys
import numpy as np
import subprocess

def run_script(input1, input2, input3):
    # Construct the command to run the main script with parameters
    python_path = sys.executable  # Get the path to the Python interpreter
    command = [python_path, "test_GMM_dirichlect_feature_weighted.py", str(input1), str(input2),str(input3)]
    
    # Print message to terminal indicating the script is being run with given parameters
    print(f"Running script with parameters: input1={input1}, input2={input2}, input3={input3}")

    # Run the command and wait for it to complete
    process = subprocess.run(command, text=True)

    # Check if the script ran successfully before proceeding to the next
    if process.returncode != 0:
        print(f"Script failed with parameters: input1={input1}, input2={input2}, input3={input3}")
        # Optionally break or continue based on your preference

# List of parameter sets you want to run the script with
parameters_list = []
for i in list(list(np.arange(0, 7,0.5))) + list(range(7, 16)):
    for j in range(4):  # This loop will iterate over [0, 1, 2, 3]
        for k in range(2):
            parameters_list.append((i, j, k))
            
# Print parameters_list to verify it has the correct values
print(parameters_list)

max = len(parameters_list)

print("Total param number: ",max)
i=0

# Loop through each parameter set and run the script
for params in parameters_list:
    run_script(*params)
    print("FINISHED FOR",params),

    i+=1

    print("============",f"  {i/max*100} % FINISHED   ","============")

        # File writing without lock
    with open("tracker.txt", 'a') as f:
        f.write(f"============  {i/max*100} % FINISHED   ============\n")