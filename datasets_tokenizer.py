# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from tqdm import tqdm
import textgrid

def execute_run():

    # Load all text files
    text_files = [] 
    text_files += list(Path("./external_datasets/librilight-processed").glob("*/*/*.txt"))
    text_files += list(Path("./external_datasets/librilight-medium-processed").glob("*/*/*.txt"))

    # Create directories
    Path("datasets").mkdir(parents=True, exist_ok=True)

    # Open output file
    with open("datasets/tokenizer.txt", "w") as tk:

        # Indexing files
        print("Build file index...")
        for file in tqdm(text_files):

            # Read file
            with open(file, "r") as f:
                lines = f.readlines()

            # Write to output
            for line in lines:
                tk.write(line.strip() + "\n")
        
    # End
    print("Done")

if __name__ == "__main__":
    execute_run()