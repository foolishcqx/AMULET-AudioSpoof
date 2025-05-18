import os
import random
import warnings
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from aug_wav import complex_randomaug  

warnings.filterwarnings('ignore')


def distribute_audio_files(input_audio_dir, num_combinations):
    """
    Evenly distribute audio files across different (Noise, Filter) combinations.
    
    Args:
        input_audio_dir: Path to the input audio directory
        num_combinations: Number of combinations (fixed at 21)
    
    Returns:
        Dictionary mapping each combination ID to a list of audio files
    """
    # Get all input audio files
    audio_files = [
        os.path.join(input_audio_dir, f) 
        for f in os.listdir(input_audio_dir) 
        if f.endswith((".wav", ".flac"))
    ]

    # Shuffle audio files to ensure even distribution
    random.shuffle(audio_files)

    # Initialize dictionary to store assignments
    assigned_files = {combo_id: [] for combo_id in range(1, num_combinations + 1)}

    # Distribute files in round-robin fashion
    for i, audio_file in enumerate(audio_files):
        combo_id = (i % num_combinations) + 1  # Ensure IDs start from 1
        assigned_files[combo_id].append(audio_file)

    return assigned_files


def process_audio_files(audio_files, output_folder, log_file, noise_id, filter_id, 
                        is_train=False, apply_filter_first=None):
    """
    Process audio files with augmentation and save the results.

    Args:
        audio_files: List of audio files to process
        output_folder: Output directory for augmented audio
        log_file: Path to log file for recording augmentation details
        noise_id: ID of the noise augmentation method
        filter_id: ID of the filter augmentation method
        is_train: Whether processing training set
        apply_filter_first: Whether to apply filter before noise
    """
    for audio_file in tqdm(audio_files, desc=f"Processing Noise {noise_id} - Filter {filter_id}"):
        # Load audio file
        audio, sr = librosa.load(audio_file, sr=16000)

        # Apply augmentation with specified noise and filter methods
        augmented_audio, _, _ = my_complex_randomaug(
            audio, 
            is_train=is_train, 
            noise_method=noise_id, 
            filter_method=filter_id, 
            apply_filter_first=apply_filter_first
        )

        # Generate output file path
        filename = os.path.basename(audio_file).split('.')[0]
        output_path = os.path.join(output_folder, f"{filename}.flac")
        
        # Save augmented audio
        sf.write(output_path, augmented_audio, sr)

        # Log augmentation information
        log_info = f"{filename} Noise:{noise_id} Filter:{filter_id}\n"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_info)


def main():
    """Main function to orchestrate the audio augmentation process."""
    # Define input/output paths
    input_folder = "/data/zhongjiafeng/ASVspooft2019_LA_2021_eval/ASVspoof2019_LA_eval/flac/"
    output_folder_base = "/data/zhongjiafeng/LA_MoE/SSL_Anti-spoofing/data/Noise_and_Filter"
    output_logs_dir = "/data/zhongjiafeng/LA_MoE/SSL_Anti-spoofing/data/"

    # Ensure output directory exists
    os.makedirs(output_logs_dir, exist_ok=True)

    # Set number of combinations (3 noise types × 7 filter types = 21 combinations)
    num_combinations = 21

    # Generate all (noise_id, filter_id) combinations
    noise_filter_combinations = [(n, f) for n in range(1, 4) for f in range(1, 8)]

    # Distribute audio files evenly across combinations
    audio_files_dict = distribute_audio_files(input_folder, num_combinations)

    # Process with both filter-first and noise-first approaches
    filter_first_options = [False, True]

    for apply_filter_first in filter_first_options:
        # Create output folder based on processing order
        folder_name = "FilterFirst" if apply_filter_first else "NoiseFirst"
        output_folder = os.path.join(output_folder_base, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # Create log file
        output_log_path = os.path.join(output_logs_dir, f"{folder_name}.txt")

        # Process each combination
        for combo_id, audio_files in audio_files_dict.items():
            noise_id, filter_id = noise_filter_combinations[combo_id - 1]  # combo_id is 1-based
            process_audio_files(
                audio_files, 
                output_folder, 
                output_log_path, 
                noise_id=noise_id, 
                filter_id=filter_id, 
                is_train=False, 
                apply_filter_first=apply_filter_first
            )


if __name__ == "__main__":
    main()
