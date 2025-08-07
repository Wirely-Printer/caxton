import os
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def move_files_from_inner_to_main(dir_path):
    """
    Move files from inner redundant folders to the main folder and delete the inner folder if empty.
    
    Args:
        dir_path (str): The path to the main directory containing subdirectories.
    """
    # Get a list of all subdirectories in the specified path
    subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

    for subdir in subdirs:
        main_subdir_path = os.path.join(dir_path, subdir)
        inner_subdir_path = os.path.join(main_subdir_path, subdir)

        # Check if the inner subdirectory exists
        if os.path.isdir(inner_subdir_path):
            logging.info(f"Processing {inner_subdir_path}...")

            # Move all files from the inner folder to the main subfolder
            files_moved = 0
            for file_name in os.listdir(inner_subdir_path):
                src_file = os.path.join(inner_subdir_path, file_name)
                dest_file = os.path.join(main_subdir_path, file_name)

                # Move the file
                try:
                    shutil.move(src_file, dest_file)
                    files_moved += 1
                    logging.info(f"Moved file {file_name} to {main_subdir_path}")
                except Exception as e:
                    logging.error(f"Failed to move {file_name} from {inner_subdir_path} to {main_subdir_path}: {e}")

            # Check if all files were moved and delete the inner folder if it's empty
            if not os.listdir(inner_subdir_path):
                try:
                    os.rmdir(inner_subdir_path)
                    logging.info(f"Deleted empty folder {inner_subdir_path}")
                except Exception as e:
                    logging.error(f"Failed to delete folder {inner_subdir_path}: {e}")

            logging.info(f"Moved {files_moved} files from {inner_subdir_path} to {main_subdir_path}")
        else:
            logging.info(f"No redundant inner folder in {main_subdir_path}. Skipping.")

# Specify the directory path here
dir_path = "D:/data"
move_files_from_inner_to_main(dir_path)
