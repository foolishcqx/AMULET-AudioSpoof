import glob
import librosa
import numpy as np
import random
import os
import soundfile as sf
from tqdm import tqdm
import warnings
from aug_wav import *

# Suppress warnings
warnings.filterwarnings('ignore')

def distribute_audio_files(input_audio_dir, kinds):
    """
    Evenly distribute audio files to each codec, ensuring similar number of files per method.
    
    Args:
        input_audio_dir: Path to the input audio directory
        kinds: Number of different filter types to use
    
    Returns:
        Dictionary mapping each filter type to its corresponding audio file list
    """
    # Get all input audio files
    audio_files = [
        os.path.join(input_audio_dir, f) 
        for f in os.listdir(input_audio_dir) 
        if f.endswith((".wav", ".flac"))
    ]

    # Calculate how many files should be assigned to each format
    num_formats = kinds
    num_files = len(audio_files)
    files_per_format = num_files // num_formats  # Number of files per format

    # Shuffle audio files to ensure even distribution
    random.shuffle(audio_files)

    # Assign audio files to each filter type
    assigned_files = {format_name: [] for format_name in range(kinds)}

    for i, audio_file in enumerate(audio_files):
        # Ensure each format has roughly the same number of files
        format_name = i % num_formats
        assigned_files[format_name].append(audio_file)

    return assigned_files


def process_audio_files(input_folder, output_folder, log_files, agument_metthod, is_train):
    """
    Apply random audio augmentation to all audio files in the input folder.
    
    Args:
        input_folder: List of input audio files
        output_folder: Directory to save augmented audio
        log_files: Path to log file for recording augmentation details
        agument_metthod: Augmentation method ID to use
        is_train: Whether processing training set
    """
    for audio_file in tqdm(input_folder):
        # Load audio file
        audio, sr = librosa.load(audio_file, sr=16000)
        
        # Apply augmentation
        augmented_audio, augmented_option = apply_Filter(
            audio, 
            is_train=is_train, 
            augument_method=agument_metthod
        )
        
        # Save augmented audio to output folder
        filename = os.path.basename(audio_file).split('.')[0]
        output_path = os.path.join(output_folder, f"{filename}.flac")
        sf.write(output_path, augmented_audio, sr)
        
        # Log augmentation information
        information = f"{filename} {augmented_option}\n"
        with open(log_files, 'a', encoding='utf-8') as f:
            f.write(information)


def main():
    """Main function to orchestrate the audio augmentation process."""
    # Define input/output paths
    folder_paths = [
        "../ASVspoof2019_LA_train/flac",
        "../ASVspoof2019_LA_dev/flac",
        "../ASVspoof2019_LA_eval/flac"
    ]
    
    output_folders = [
        "../data/2019_LA_train_Filter",
        "../data/2019_LA_dev_Filter",
        "../data/2019_LA_eavl_Filter"
    ]
    
    output_logs_dir = "../data/"
    
    # Configuration for each dataset
    train_ornot = [True, True, False]  # Training flag for each dataset
    n_kinds = [3, 3, 7]  # Number of filter types for each dataset
    
    # Process each dataset
    for folder_path, output_folder, is_train, kinds in zip(folder_paths, output_folders, train_ornot, n_kinds):
        # Ensure output directories exist
        os.makedirs(output_logs_dir, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        
        # Create log file path
        output_logs_path = os.path.join(output_logs_dir, f"{os.path.basename(output_folder)}.txt")
        
        # Distribute audio files across filter types
        audio_files_dict = distribute_audio_files(folder_path, kinds)
        
        # Process each group of files with its assigned filter
        for i, audio_files in audio_files_dict.items():
            process_audio_files(
                audio_files, 
                output_folder, 
                output_logs_path, 
                agument_metthod=i, 
                is_train=is_train
            )


if __name__ == "__main__":
    main()