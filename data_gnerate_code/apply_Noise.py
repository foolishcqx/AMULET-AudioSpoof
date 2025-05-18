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
    Evenly distribute audio files across different noise types, ensuring similar number of files per method.
    
    Args:
        input_audio_dir: Path to the input audio directory
        kinds: Number of different noise types to use
    
    Returns:
        Dictionary mapping each noise type to its corresponding audio file list
    """
    # Get all input audio files
    audio_files = [
        os.path.join(input_audio_dir, f) 
        for f in os.listdir(input_audio_dir) 
        if f.endswith((".wav", ".flac"))
    ]

    # Calculate distribution
    num_formats = kinds
    num_files = len(audio_files)
    
    # Shuffle audio files to ensure even distribution
    random.shuffle(audio_files)

    # Assign audio files to each noise type
    assigned_files = {format_id: [] for format_id in range(kinds)}

    for i, audio_file in enumerate(audio_files):
        # Ensure each format has roughly the same number of files
        format_id = i % num_formats
        assigned_files[format_id].append(audio_file)

    return assigned_files


def process_audio_files(audio_files, output_folder, log_file, augment_method, is_train):
    """
    Apply noise augmentation to audio files and save the results.
    
    Args:
        audio_files: List of audio files to process
        output_folder: Directory to save augmented audio
        log_file: Path to log file for recording augmentation details
        augment_method: Noise augmentation method ID to use
        is_train: Whether processing training set
    """
    for audio_file in tqdm(audio_files, desc=f"Processing noise method {augment_method}"):
        # Load audio file
        audio, sr = librosa.load(audio_file, sr=16000)
        
        # Apply noise augmentation
        augmented_audio, augmented_option = apply_Noise(
            audio, 
            is_train=is_train, 
            augument_method=augment_method
        )
        
        # Save augmented audio to output folder
        filename = os.path.basename(audio_file).split('.')[0]
        output_path = os.path.join(output_folder, f"{filename}.flac")
        sf.write(output_path, augmented_audio, sr)
        
        # Log augmentation information
        information = f"{filename} {augmented_option}\n"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(information)


def main():
    """Main function to orchestrate the audio noise augmentation process."""
    # Define input/output paths
    folder_paths = [
        "/data/zhongjiafeng/ASVspooft2019_LA_2021_eval/ASVspoof2019_LA_eval/flac"
    ]
    
    output_folders = [
        "/data/zhongjiafeng/LA_MoE/SSL_Anti-spoofing/data_noise/unseen_Noise_high"
    ]
    
    output_logs_dir = "/data/zhongjiafeng/LA_MoE/SSL_Anti-spoofing/data_noise/"
    
    # Configuration for each dataset
    train_flags = [False]  # Training flag for each dataset
    noise_types = [3]      # Number of noise types for each dataset
    
    # Process each dataset
    for folder_path, output_folder, is_train, kinds in zip(folder_paths, output_folders, train_flags, noise_types):
        # Ensure output directories exist
        os.makedirs(output_logs_dir, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        
        # Create log file path
        output_log_path = os.path.join(output_logs_dir, f"{os.path.basename(output_folder)}.txt")
        
        # Distribute audio files across noise types
        audio_files_dict = distribute_audio_files(folder_path, kinds)
        
        # Process each group of files with its assigned noise type
        for noise_id, audio_files in audio_files_dict.items():
            process_audio_files(
                audio_files, 
                output_folder, 
                output_log_path, 
                augment_method=noise_id, 
                is_train=is_train
            )


if __name__ == "__main__":
    main()

