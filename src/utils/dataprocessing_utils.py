import os
import pandas as pd
import h5py
import numpy as np

def get_train_folders(root_dir):
    """Returns a sorted list of train_X directories."""
    return sorted([f for f in os.listdir(root_dir) if f.startswith("train_") and os.path.isdir(os.path.join(root_dir, f))])

def merge_metadata(train_folders, output_metadata_path):
    """Merges all metadata.csv files into a single file."""
    metadata_list = []
    
    for folder in train_folders:
        metadata_file = os.path.join(folder, "metadata.csv")
        print(f"Checking: {metadata_file}")  # Debug print
        if os.path.exists(metadata_file):
            df = pd.read_csv(metadata_file)
            metadata_list.append(df)

    if metadata_list:
        combined_metadata = pd.concat(metadata_list, ignore_index=True)
        combined_metadata.to_csv(output_metadata_path, index=False)
        print(f"Saved combined metadata to {output_metadata_path}")
    else:
        print("No metadata.csv files found.")

def merge_hdf5(train_folders, output_h5_path):
    """Merges all image_data.h5 files into a single file."""
    first_file = True
    with h5py.File(output_h5_path, "w") as output_h5:
        for folder in train_folders:
            h5_file = os.path.join(folder, "image_data.h5")
            if os.path.exists(h5_file):
                with h5py.File(h5_file, "r") as h5_input:
                    for dataset_name in h5_input.keys():
                        data = h5_input[dataset_name][:]
                        
                        if first_file:
                            output_h5.create_dataset(dataset_name, data=data, maxshape=(None, *data.shape[1:]), chunks=True)
                        else:
                            dset = output_h5[dataset_name]
                            dset.resize((dset.shape[0] + data.shape[0]), axis=0)
                            dset[-data.shape[0]:] = data
                first_file = False
    print(f"Saved combined image data to {output_h5_path}")

def main(root_dir):
    """Main function to merge all train_X folders into a single train folder."""
    # root_dir = "../datasets/"  # Change this if needed
    output_dir = os.path.join(root_dir, "train")
    os.makedirs(output_dir, exist_ok=True)

    train_folders = get_train_folders(root_dir)
    output_metadata_path = os.path.join(output_dir, "metadata.csv")
    output_h5_path = os.path.join(output_dir, "image_data.h5")

    merge_metadata(train_folders, output_metadata_path)
    merge_hdf5(train_folders, output_h5_path)

if __name__ == "__main__":
    main()
