import os
import json
import shutil

""" Script to download data sets from mounted drive T.
"""

def create_file_mapping(folder_path):
    file_mapping = {}
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith("batch_") and filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                # Read each line in the file
                for line in file:
                    line = line.strip().replace("_ts.nc","")  # Remove leading/trailing whitespace and newlines
                    if line:  # Ensure line is not empty
                        file_mapping[line] = filename.replace(".txt","")
    
    # Write the mapping to a JSON file
    output_file = "article_data/file_mapping.json"
    with open(output_file, 'w') as json_file:
        json.dump(file_mapping, json_file, indent=4)
    
    print(f"File mapping created successfully in {output_file}")

def download_dataset(filename, from_folder, to_folder, max_number_of_scenarios = None):
    with open("/home/ebr/projects/inundation-emulator/article_data/file_mapping.json") as file:
        filemap = json.load(file)
    scenarios = []
    with open(filename, 'r') as file:
        # Read each line in the file
        for scenario_nr, line in enumerate(file):
            line = line.strip()
            scenario = line.split("/")[1]
            scenarios.append(scenario + "\n")
            print(os.path.join(filemap[line], scenario))
            sub_dir_from_folder = os.path.join(from_folder, filemap[line]) 
            for f in os.listdir(sub_dir_from_folder):
                if f.startswith(line.split("/")[1]):
                    print(f)
                    shutil.copy(os.path.join(sub_dir_from_folder, f), to_folder)
            if max_number_of_scenarios and scenario_nr+1 >= max_number_of_scenarios:
                break

    with open(os.path.join(to_folder, "scenarios.txt"), "w") as f:
        f.writelines(scenarios)

# Example usage
def main():
    # Source folder
    folder_path = "/mnt/NGI_disks/ebr/T/Tsunami/PTHA2020_runs_UMA"
    # Local folder
    #to_folder = "/home/ebr/data/PTHA2020_runs_UMA/train_591"
    to_folder = "/home/ebr/data/PTHA2020_runs_UMA/test"
    
    create_file_mapping(folder_path)
    # Scenarios (to copy)
    #filename="/home/ebr/projects/inundation-emulator/article_data/train_591/train.txt"
    filename="/home/ebr/projects/inundation-emulator/article_data/bottom_UMAPS_shuf.txt"
    
    download_dataset(filename,
                     folder_path,
                     to_folder,
                     max_number_of_scenarios=100)
    
if __name__ == "__main__":
    main()