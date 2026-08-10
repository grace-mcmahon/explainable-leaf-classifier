"""
1_prepare_data.py
...docstring...
"""
#This is triple-quoted tect located at the top of the file. This is just documentation so that as soon as I open this file i know the purpose of this code

import os # this import is for interacting with the operating system 
import shutil #'shell utilities' specifically for compying files 
import random # for shuffling things randomly so that we dont have bias by file order in our train/val/test split 
import pathlib 
import Path # a modern way to handle file paths 

RAW_DATA_DIR = Path("data/raw") # name of the input file of data 
OUTPUT_DIR = Path("data")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1 
SEED = 42
# these are all constants. TRAIN_RATIO = 0.8 means 80% of images go to training 
# SEED = 42 is important as it makes the 'random' shuffle reproducible therefore the same seed always rpoduces the same shiffle order and the results end up beign consistent everytime you run the code 

def main(): # def defines the function so everytime we call main what happens inside of this block will run. Main is given as the 'function that runs everything'
    random.seed(SEED) # this tells Python's random number generator to start from a fixed point (42), so random.shuffle() later gives the same result every time you run the script 
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(...)
    # this is a safety check: if the folder does not exist yet (this may be the case if you ahve not downloaded the dataset), stop imeediately witha  clear error message, rather than failing and leaving us having to debug more complex, longer code segment 
    class_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()] # This is what we call a list comprehension. This is just a more compact way of writing a loop that builds a list. In essence we look at eveyr item inside RAW_DATA_DIR, and keep only the ones that are folders (not files). So each clss (e.g. Tomato__healthy) is its own sub-folder in the downloaded dataset so we can collect all your classes into a list.
    print(f"Found {class_dirs} classes")
    #f"..." is an f-string so it will elt you insert a variable's valud directly into text using {}. len(class_dirs) counts how many items are in that list. This just prints a helpful status message to the terminal 
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)
        # A loop over three strings. OUTPUT_DIR / split uses Path's neat trick - the / operator join paths, so this becomes data/train, data/val, data/test..mkdir() creates the folder; exist_ok = TRUE means 'dont error if it alreayd exists' 
        for class_dir in class_dirs:
            images = list(class_dir.glob("*.*"))
            random.shuffle(images)
            # We loop over eahc class .glob("*.*") finds every file inside it with any extension (the images). list(...) converts that into an actual list we cna work with. random.shuufle() mixes up the order - important so the split isnt accidently biased (e.g. all the blurry photos happening at the end.)
            n_total = len(images)
            n_train = int(n_total * TRAIN_RATIO)
            n_val = int(n_total * VAL_RATIO)
            # Basic arithmetic: total image count, then how mnay go to train(80%) and val(10%). int(...) round down to a whole number since you cant have a fraction of an image.
            splits = {
                "train" : images[:n_train],
                "val": images[n_train:n_train + n_val],
                "test": images[n_train + n_val:],
            }
            # This is a dictionary mapping each split name to a slice of the shuffled image list. images[:n_train] means 'everything from the start up to n_train' (python splicing)
            # images[n_train:n_train + n_val] is the next chunk
            # images[n_train + n_Val:] is everything ledt over - this becomes the test set automatically 
            for split_name, split_images in splits.items():
                dest_dir = OUTPUT_DIR / split_name / class_dir.name
                dest_dir.mkdir(parent=True, exist_ok=True)
                for img_path in split_images:
                    shutil.copy(img_path, dest_dir / img_path.name)
                    # .items() lets us loop over the dictionaries key-value pairs together. For each split, we build the destination folder path (e.g. data/train/Tomato__healthy), create it, copy eveyr image into it one at a time with shutil.copy
            print(f"{class_dir.name}: {n_total} images -> ...")
                    # This is just a status update per class, so that you can watch progress in the temrianl as it runs
        print("\nDone. Data organised under data/train, data/val, data/test")
                #\n is a newline character inserting a blank line before the text, readability purposes in the terminal output 
            
if __name__ == "__main__":
    main()
# A common python pattern. It is basically saying only run this file directly i.e. 'main()' when someone runs this script directly. its a safety convention thing     

 