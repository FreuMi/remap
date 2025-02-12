import subprocess
import os
import sys

total_cnt = 0
passed_cnt = 0

def preprocess_line(line):
    # Replace _: values with a consistent placeholder, since burp generates a unique id
    return ' '.join(
        'BLANK_NODE' if part.startswith('_:') else part for part in line.split()
    )

def compare_files(file1, file2):
    # Read and preprocess lines from the first file into a set
    print("Loading", file1)
    with open(file1, 'r') as f1:
        lines1 = set(preprocess_line(line.strip()) for line in f1)

    print("Loading", file2)
    # Read and preprocess lines from the second file into a set
    with open(file2, 'r') as f2:
        lines2 = set(preprocess_line(line.strip()) for line in f2)

    # Compare the two sets
    if lines1 == lines2:
        return True
    else:
        print("The files do not have the same lines.")
        # Print differences
        print("Lines only in the first file:", lines1 - lines2)
        print("Lines only in the second file:", lines2 - lines1)
        sys.exit()

def run_console_program(command):
    try:
        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Print the output of the command
        # print("Output:\n", result.stdout)
        if result.stderr:
            print("Errors:\n", result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print("Command output:\n", e.output)

def get_folder_names(directory):
    # List all items in the directory and filter only folders
    folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    
    # Sort the folder names alphabetically
    return sorted(folders)


directory = './csv_test_cases/'
folders = get_folder_names(directory)

total_cnt = len(folders)

for folder in folders:
    if "CSV" not in folder:
        continue
    print(f"Working on {folder}...")
    all_files = os.listdir(f'{directory}{folder}')
    
    # Get csv and rdf file
    csv_files = []
    rdf_file = ""
    for file in all_files:
        if ".csv" in file:
            csv_files.append(file)
        elif ".nq" in file:
            rdf_file = file

    csv_file_strs = [f'{directory}{folder}/{csv_file}' for csv_file in csv_files]

    # Clear output
    with open("res.nq", "w") as file:
        pass


    rdf_file_refernce = f'{directory}{folder}/output.nq'
    command = ['python3', 'remap.py', '--csv'] + csv_file_strs + ["--rdf", f'{directory}{folder}/{rdf_file}']
    run_console_program(command)

    print("===")
    # Execute generate mapping file
    command = ["java", "-jar", "burp.jar", "-m", "generated_mapping.ttl", "-o", "res.nq", "-b", "http://example.com/base/"]
    run_console_program(command)

    # Compare outputs 
    res = compare_files("./res.nq", f'{directory}{folder}/output.nq')
    if res == True:
        passed_cnt += 1
        print("Passed.")
        print("=============")
        
print("=============")
print(f"{passed_cnt} of {total_cnt} passed successful!")