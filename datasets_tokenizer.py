# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from tqdm import tqdm
import textgrid

def execute_run():

    # Load all text files
    ids = []
    for dataset in ["librilight-large-processed"]:
        with open("./external_datasets/" + dataset + "/files_valid.txt", 'r') as file:
            lines = file.readlines()
        ids += [(dataset + "/" + l.strip()) for l in lines]
    ids.sort()

    # Create directories
    Path("datasets").mkdir(parents=True, exist_ok=True)

    # Open output file
    with open("datasets/tokenizer.txt", "w") as tk:

        # Indexing files
        print("Build file index...")
        for id in tqdm(ids):

            # Read file
            with open(id + ".txt", "r") as f:
                lines = f.readlines()

            # Write to output
            for line in lines:
                tk.write(line.strip() + "\n")
        
    # End
    print("Done")

if __name__ == "__main__":
    execute_run()