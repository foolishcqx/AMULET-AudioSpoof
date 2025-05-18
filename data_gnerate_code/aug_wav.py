import glob
import random

import librosa
import numpy as np
from audiomentations import *

# from Script.data_padding import tensor_padding_1d


###  Time-domain augmentation ---- Masking  ### 
def wav_time_mask(input_data, wav_samp_rate):
    """ output = wav_time_mask(input_data, wav_samp_rate)
    
    Apply time mask and zero-out segments
    
    input
    -----
      input_data: np.array, (length, 1)
      wav_samp_rate: int, waveform sampling rate
    
    output
    ------
      output:  np.array, (length, 1)
    """
    mask_rate =  random.uniform(0.2, 0.5)
    # choose the codec
    seg_width = int(np.random.rand() * mask_rate * wav_samp_rate)
    start_idx = int(np.random.rand() * (input_data.shape[0] - seg_width))
    if start_idx < 0:
        start_idx = 0
    if (start_idx + seg_width) > input_data.shape[0]:
        seg_width = input_data.shape[0] - start_idx
    tmp = np.ones_like(input_data)
    tmp[start_idx:start_idx+seg_width] = 0
    return input_data * tmp

###  Time-domain augmentation ---- codec  ### 
def mulaw_encode(x, quantization_channels, scale_to_int=True):
    """x_mu = mulaw_encode(x, quantization_channels, scale_to_int=True)
    
    mu-law companding

    input
    -----
       x: np.array, float-valued waveforms in (-1, 1)
       quantization_channels (int): Number of channels
       scale_to_int: Bool
         True: scale mu-law to int
         False: return mu-law in (-1, 1)

    output
    ------
       x_mu: np.array, mulaw companded wave
    """
    mu = quantization_channels - 1.0
    x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    if scale_to_int:
        x_mu = np.array((x_mu + 1) / 2 * mu + 0.5, dtype=np.int32)
    return x_mu

def mulaw_decode(x_mu, quantization_channels, input_int=True):
    """mulaw_decode(x_mu, quantization_channels, input_int=True)
    
    mu-law decoding

    input
    -----
      x_mu: np.array, mu-law waveform
      quantization_channels: int, Number of channels
      input_int: Bool
        True: convert x_mu (int) from int to float, before mu-law decode
        False: directly decode x_mu (float)

    output
    ------
        x: np.array, waveform from mulaw decoding
    """
    mu = quantization_channels - 1.0
    if input_int:
        x = x_mu / mu * 2 - 1.0
    else:
        x = x_mu
    x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.0) / mu
    return x

def load_mulaw(x):
    x_mu_encoded = mulaw_encode(x, quantization_channels=256, scale_to_int=True)
    x_mu_decoded = mulaw_decode(x_mu_encoded, quantization_channels=256, input_int=True)
    return x_mu_decoded

def alaw_encode(x, quantization_channels, scale_to_int=True, A=87.6):
    """x_a = alaw_encoder(x, quantization_channels, scale_to_int=True, A=87.6)

    input
    -----
       x: np.array, float-valued waveforms in (-1, 1)
       quantization_channels (int): Number of channels
       scale_to_int: Bool
         True: scale mu-law to int
         False: return mu-law in (-1, 1)
       A: float, parameter for a-law, default 87.6
    output
    ------
       x_a: np.array, a-law companded waveform
    """
    num = quantization_channels - 1.0
    x_abs = np.abs(x)
    flag = (x_abs * A) >= 1
    
    x_a = A * x_abs
    x_a[flag] = 1 + np.log(x_a[flag])
    x_a = np.sign(x) * x_a / (1 + np.log(A))
    
    if scale_to_int:
        x_a = np.array((x_a + 1) / 2 * num + 0.5, dtype=np.int32)
    return x_a

def alaw_decode(x_a, quantization_channels, input_int=True, A=87.6):
    """alaw_decode(x_a, quantization_channels, input_int=True)

    input
    -----
      x_a: np.array, mu-law waveform
      quantization_channels: int, Number of channels
      input_int: Bool
        True: convert x_mu (int) from int to float, before mu-law decode
        False: directly decode x_mu (float)
       A: float, parameter for a-law, default 87.6
    output
    ------
       x: np.array, waveform
    """
    num = quantization_channels - 1.0
    if input_int:
        x = x_a / num * 2 - 1.0
    else:
        x = x_a
        
    sign = np.sign(x)
    x_a_abs = np.abs(x)
    
    x = x_a_abs * (1 + np.log(A))
    flag = x >= 1
    
    x[flag] = np.exp(x[flag] - 1)
    x = sign * x / A
    return x

def load_alaw(x):
    x_a_encoded = alaw_encode(x, quantization_channels=256, scale_to_int=True, A=87.6)
    x_a_decoded = alaw_decode(x_a_encoded, quantization_channels=256, input_int=True, A=87.6)
    return x_a_decoded

###  Time-domain augmentation ---- Random Aug ### 
def randomaug(x, is_train = False, augument_method = False):
    # 定义 p_random
    #p_random = random.choice([0.1, 0.5, 1.0])
    p_random = 1.0
    # 定义所有的增强操作和它们的编号
    if is_train:
        noise_type = [
            (1, AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=p_random)),
            (2, AirAbsorption(min_distance=10, max_distance=50, p=p_random)),
            (3, Aliasing(min_sample_rate=4000, max_sample_rate=8000, p=p_random)),
            (4, BandPassFilter(min_center_freq=100, max_center_freq=7500, p=p_random)),
            (5, Shift(p=p_random)),
            (6, PitchShift(min_semitones=-4, max_semitones=4, p=p_random)),
            (7, HighPassFilter(min_cutoff_freq=100, max_cutoff_freq=7500, p=p_random)),
            (8, Mp3Compression(min_bitrate=8, max_bitrate=64, p=p_random)),
            # (9, LowPassFilter(min_cutoff_freq=100, max_cutoff_freq=7500, p=p_random)),
            # (10, Limiter(min_threshold_db=-16.0, max_threshold_db=-6.0, threshold_mode='relative_to_signal_peak', p=p_random)),
            # (11, PolarityInversion(p=p_random)),
            # (12, PeakingFilter(min_center_freq=100, max_center_freq=7500, p=p_random)),
            # (13, TimeStretch(min_rate=0.8, max_rate=1.25, p=p_random)),
            # (14, TimeMask(min_band_part=0.1, max_band_part=0.5, fade=True, p=p_random)),
            # (15, TanhDistortion(min_distortion=0.01, max_distortion=0.7, p=p_random))
        ]
    else:
        noise_type = [
        (1, AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=p_random)),
        (2, AirAbsorption(min_distance=10, max_distance=50, p=p_random)),
        (3, Aliasing(min_sample_rate=4000, max_sample_rate=8000, p=p_random)),
        (4, BandPassFilter(min_center_freq=100, max_center_freq=7500, p=p_random)),
        (5, Shift(p=p_random)),
        (6, PitchShift(min_semitones=-4, max_semitones=4, p=p_random)),
        (7, HighPassFilter(min_cutoff_freq=100, max_cutoff_freq=7500, p=p_random)),
        (8, Mp3Compression(min_bitrate=8, max_bitrate=64, p=p_random)),
        (9, LowPassFilter(min_cutoff_freq=100, max_cutoff_freq=7500, p=p_random)),
        (10, Limiter(min_threshold_db=-16.0, max_threshold_db=-6.0, threshold_mode='relative_to_signal_peak', p=p_random)),
        (11, PolarityInversion(p=p_random)),
        (12, PeakingFilter(min_center_freq=100, max_center_freq=7500, p=p_random)),
        (13, TimeStretch(min_rate=0.8, max_rate=1.25, p=p_random)),
        (14, TimeMask(min_band_part=0.1, max_band_part=0.5, fade=True, p=p_random)),
        (15, TanhDistortion(min_distortion=0.01, max_distortion=0.7, p=p_random))
    ]
    # 随机选择一个操作
    if augument_method:
        operation_id, augment = noise_type[augument_method]
    else:
        operation_id, augment = random.choice(noise_type)
    
    # 确保 x 是 float32 类型
    x = np.array(x, dtype=np.float32)  
    
    # 执行增强操作
    aug_audio = augment(samples=x, sample_rate=16000)
    
    # 返回增强后的音频和所选操作的编号
    return aug_audio, operation_id
###  Time-domain augmentation ---- Random Aug ### 
def other_randomaug(x, is_train = False, augument_method = False):
    # 定义 p_random
    #p_random = random.choice([0.1, 0.5, 1.0])
    p_random = 1.0
    # 定义所有的增强操作和它们的编号
    if is_train:
        noise_type = [
            (1, AirAbsorption(min_distance=10, max_distance=50, p=p_random)),
            (2, Aliasing(min_sample_rate=4000, max_sample_rate=8000, p=p_random)),
            (3, Shift(p=p_random)),
            (4, PitchShift(min_semitones=-4, max_semitones=4, p=p_random)),
            (5, Mp3Compression(min_bitrate=8, max_bitrate=64, p=p_random)),
            # (9, LowPassFilter(min_cutoff_freq=100, max_cutoff_freq=7500, p=p_random)),
            # (10, Limiter(min_threshold_db=-16.0, max_threshold_db=-6.0, threshold_mode='relative_to_signal_peak', p=p_random)),
            # (11, PolarityInversion(p=p_random)),
            # (12, PeakingFilter(min_center_freq=100, max_center_freq=7500, p=p_random)),
            # (13, TimeStretch(min_rate=0.8, max_rate=1.25, p=p_random)),
            # (14, TimeMask(min_band_part=0.1, max_band_part=0.5, fade=True, p=p_random)),
            # (15, TanhDistortion(min_distortion=0.01, max_distortion=0.7, p=p_random))
        ]
    else:
        noise_type = [
        (1, AirAbsorption(min_distance=10, max_distance=50, p=p_random)),
        (2, Aliasing(min_sample_rate=4000, max_sample_rate=8000, p=p_random)),
        (3, Shift(p=p_random)),
        (4, PitchShift(min_semitones=-4, max_semitones=4, p=p_random)),
        (5, Mp3Compression(min_bitrate=8, max_bitrate=64, p=p_random)),
        (6, Limiter(min_threshold_db=-16.0, max_threshold_db=-6.0, threshold_mode='relative_to_signal_peak', p=p_random)),
        (7, PolarityInversion(p=p_random)),
        (8, TimeStretch(min_rate=0.8, max_rate=1.25, p=p_random)),
        (9, TimeMask(min_band_part=0.1, max_band_part=0.5, fade=True, p=p_random)),
        (10, TanhDistortion(min_distortion=0.01, max_distortion=0.7, p=p_random))
    ]
    # 随机选择一个操作
    if augument_method:
        operation_id, augment = noise_type[augument_method]
    else:
        operation_id, augment = random.choice(noise_type)
    
    # 确保 x 是 float32 类型
    x = np.array(x, dtype=np.float32)  
    
    # 执行增强操作
    aug_audio = augment(samples=x, sample_rate=16000)
    
    # 返回增强后的音频和所选操作的编号
    return aug_audio, operation_id

# def complex_randomaug(x):
#     # Noise Augmentation: Need Background Noise, short noise, Impulse Response noise
#     # all_type = [
#     #             AddColorNoise(min_snr_db=5.0, max_snr_db=40.0, min_f_decay=-6.0, max_f_decay=6.0, p=0.5),
#     #             AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
#     #             AddGaussianSNR(min_snr_db=5.0, max_snr_db=40.0, p=0.5),
#     #             AdjustDuration(duration_seconds=4.0, p=0.5),
#     #             AirAbsorption(min_distance=10.0, max_distance=50.0, p=0.5),
#     #             Aliasing(min_sample_rate=4000, max_sample_rate=16000, p=0.6),
#     #             BitCrush(min_bit_depth=5, max_bit_depth=14, p=0.6),
#     #             Reverse(p=0.6),
#     #             Normalize(p=0.6),
#     #             Padding(mode='silence', min_fraction=0.01, max_fraction=0.7, p=0.6),
#     #             PitchShift(min_semitones=-4, max_semitones=4, p=0.6),
#     #             PolarityInversion(p=0.6),
#     #             RepeatPart(mode="insert", p=0.6),
#     #             Resample(min_sample_rate=8000, max_sample_rate=16000, p=0.6),
#     #             Shift(min_shift=-0.5, max_shift=0.5, p=0.6),
#     #             TanhDistortion(min_distortion=0.01, max_distortion=0.7, p=0.6),   
#     #             TimeMask(min_band_part=0.2, max_band_part=0.25, fade=True, p=0.6),           
#     #             TimeStretch(min_rate=0.8, max_rate=1.25, p=0.6),
#     #             Trim(top_db=30.0, p=0.6),
#     #             BandPassFilter(min_center_freq=100, max_center_freq=7500, p=0.6),
#     #             BandStopFilter(min_center_freq=100, max_center_freq=7500, p=0.6),
#     #             HighPassFilter(min_cutoff_freq=100, max_cutoff_freq=7500, p=0.6),
#     #             HighShelfFilter(min_center_freq=300, max_center_freq=7500, p=0.6),
#     #             LowPassFilter(min_cutoff_freq=100, max_cutoff_freq=7500, p=0.6),
#     #             LowShelfFilter(min_center_freq=50, max_center_freq=4000, p=0.6),
#     #             PeakingFilter(min_center_freq=100, max_center_freq=7500, p=0.6)
#     # ] 
    
#     p_1 = random.choice([0.1, 0.5, 1.0])
#     Noise_type = [AddColorNoise(min_snr_db=5.0, max_snr_db=40.0, min_f_decay=-6.0, max_f_decay=6.0, p=p_1),
#                     AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=p_1),
#                     AddGaussianSNR(min_snr_db=5.0, max_snr_db=40.0, p=p_1)]
#     Filter_type = [BandPassFilter(min_center_freq=random.randrange(100, 401, 100), max_center_freq=random.randrange(4000, 7501, 1000), p=p_1),
#                     BandStopFilter(min_center_freq=random.randrange(100, 401, 100), max_center_freq=random.randrange(4000, 7501, 1000), p=p_1),
#                     HighPassFilter(min_cutoff_freq=random.randrange(100, 401, 100), max_cutoff_freq=random.randrange(4000, 7501, 1000), p=p_1),
#                     HighShelfFilter(min_center_freq=random.randrange(100, 401, 100), max_center_freq=random.randrange(4000, 7501, 1000), p=p_1),
#                     LowPassFilter(min_cutoff_freq=random.randrange(100, 401, 100), max_cutoff_freq=random.randrange(4000, 7501, 1000), p=p_1),
#                     LowShelfFilter(min_center_freq=random.randrange(100, 401, 100), max_center_freq=random.randrange(4000, 7501, 1000), p=p_1),
#                     PeakingFilter(min_center_freq=random.randrange(100, 401, 100), max_center_freq=random.randrange(4000, 7501, 1000), p=p_1)]
#     Noise_aug = random.choice(Noise_type)
#     # Other_aug = random.choice(Other_type)
#     Filter_aug = random.choice(Filter_type)
#     # augment = Compose([Noise_aug, Other_aug, Filter_aug])
#     # augment = Compose([Noise_aug, Other_aug])
#     # augment = Compose([Other_aug])
#     x = np.array(x, dtype=np.float32)  # 确保 x 是 float32 类型
#     aug_noise_audio = Noise_aug(samples=x, sample_rate=16000)
#     # aug_other_audio = Other_aug(samples=aug_noise_audio, sample_rate=16000)
#     aug_filter_audio = Filter_aug(samples=aug_noise_audio, sample_rate=16000)
#     return aug_filter_audio

def complex_randomaug(x, is_train=True, noise_method=None, filter_method=None, apply_filter_first=False):
    p_1 = 1.0  # Set uniform probability for augmentation
    
    # Define noise augmentation types
    if is_train:
        noise_options = [
            (1, AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=p_1)),
            (2, AddGaussianSNR(min_snr_db=5.0, max_snr_db=40.0, p=p_1)),
        ]
    else:
        noise_options = [
            (1, AddGaussianNoise(min_amplitude=0.015, max_amplitude=0.03, p=p_1)),
            (2, AddGaussianSNR(min_snr_db=2.0, max_snr_db=5.0, p=p_1)),
            (3, AddColorNoise(min_snr_db=5.0, max_snr_db=40.0, min_f_decay=-6.0, max_f_decay=6.0, p=p_1)),#3.0,30
        ]
    
    # Select noise augmentation method (if not specified, choose randomly)
    if noise_method is not None:
        noise_id, noise_augment = noise_options[noise_method - 1]
    else:
        noise_id, noise_augment = random.choice(noise_options)

    # Define filter augmentation types (operation ID, augmentation function)
    if is_train:
        filter_options = [
            (1, BandPassFilter(min_center_freq=random.randint(100, 400), max_center_freq=random.randint(4000, 7500), p=p_1)),
            (2, HighPassFilter(min_cutoff_freq=random.randint(100, 400), max_cutoff_freq=random.randint(4000, 7500), p=p_1)),
            (3, LowPassFilter(min_cutoff_freq=random.randint(100, 400), max_cutoff_freq=random.randint(4000, 7500), p=p_1)),
        ]
    else:
        filter_options = [
            (1, BandPassFilter(min_center_freq=random.randint(100, 400), max_center_freq=random.randint(4000, 7500), p=p_1)),
            (2, HighPassFilter(min_cutoff_freq=random.randint(100, 400), max_cutoff_freq=random.randint(4000, 7500), p=p_1)),
            (3, LowPassFilter(min_cutoff_freq=random.randint(100, 400), max_cutoff_freq=random.randint(4000, 7500), p=p_1)),
            (4, BandStopFilter(min_center_freq=random.randint(100, 400), max_center_freq=random.randint(4000, 7500), p=p_1)),
            (5, HighShelfFilter(min_center_freq=random.randint(100, 400), max_center_freq=random.randint(4000, 7500), p=p_1)),
            (6, LowShelfFilter(min_center_freq=random.randint(100, 400), max_center_freq=random.randint(4000, 7500), p=p_1)),
            (7, PeakingFilter(min_center_freq=random.randint(100, 400), max_center_freq=random.randint(4000, 7500), p=p_1)),
        ]

    # Select filter augmentation method (if not specified, choose randomly)
    if filter_method is not None:
        filter_id, filter_augment = filter_options[filter_method - 1]
    else:
        filter_id, filter_augment = random.choice(filter_options)

    # Ensure input data is float32
    x = np.array(x, dtype=np.float32)

    # Process augmentation order
    if apply_filter_first:
        # Apply filter first, then add noise
        augmented_audio = filter_augment(samples=x, sample_rate=16000)
        augmented_audio = noise_augment(samples=augmented_audio, sample_rate=16000)
    else:
        # Apply noise first, then filter (default)
        augmented_audio = noise_augment(samples=x, sample_rate=16000)
        augmented_audio = filter_augment(samples=augmented_audio, sample_rate=16000)

    return augmented_audio, noise_id, filter_id

# def lod_rir_noise(x, cut_length):
def load_rir_noise(x):
    # noise = random.choice(glob.glob(self.noise_path + '*/*/*.wav'))
    noise = random.choice(glob.glob('/data/xuyx/Datasets/RIRS_NOISES/' + '/*/*/*/*.wav'))
    noise, _ = librosa.load(noise, sr=16000)
    cut_length = len(x)
    # noise = tensor_padding_1d(noise, cut_length)
    snr = random.uniform(0.2, 0.8)
    x_noise = x + snr * noise
    return x_noise

def apply_Noise(x, is_train = True, augument_method = False):
    p_1 = 1.0
    
    # 定义噪声类型列表
    if is_train:
        Noise_type = [
        (1, AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=p_1)),
        (2, AddGaussianSNR(min_snr_db=5.0, max_snr_db=40.0, p=p_1)),
        ]
    else:
        Noise_type = [
        (1, AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=p_1)),
        (2, AddGaussianSNR(min_snr_db=5.0, max_snr_db=40.0, p=p_1)),
        (3, AddColorNoise(min_snr_db=5.0, max_snr_db=40.0, min_f_decay=-6.0, max_f_decay=6.0, p=p_1)),
        ]
    
    # 随机选择一个噪声增强方法
    if augument_method:
        noise_id, Noise_aug = Noise_type[augument_method]
    else:
        noise_id, Noise_aug = random.choice(Noise_type)
    
    # 确保x是float32类型
    x = np.array(x, dtype=np.float32)
    
    # 应用噪声增强
    aug_noise_audio = Noise_aug(samples=x, sample_rate=16000)
    
    # 返回增强后的音频以及使用的噪声增强方法ID
    return aug_noise_audio, noise_id


def apply_Filter(x, is_train = True, augument_method = False):
    p_1 = 1.0
    if is_train:
        Filter_type = [(1,BandPassFilter(min_center_freq=random.randrange(100, 401, 100), max_center_freq=random.randrange(4000, 7501, 1000), p=p_1)),
                        (2,HighPassFilter(min_cutoff_freq=random.randrange(100, 401, 100), max_cutoff_freq=random.randrange(4000, 7501, 1000), p=p_1)),
                        (3,LowPassFilter(min_cutoff_freq=random.randrange(100, 401, 100), max_cutoff_freq=random.randrange(4000, 7501, 1000), p=p_1))]
    else:
        Filter_type = [(1,BandPassFilter(min_center_freq=random.randrange(100, 401, 100), max_center_freq=random.randrange(4000, 7501, 1000), p=p_1)),
                        (2,HighPassFilter(min_cutoff_freq=random.randrange(100, 401, 100), max_cutoff_freq=random.randrange(4000, 7501, 1000), p=p_1)),
                        (3,LowPassFilter(min_cutoff_freq=random.randrange(100, 401, 100), max_cutoff_freq=random.randrange(4000, 7501, 1000), p=p_1)),
                        (4,BandStopFilter(min_center_freq=random.randrange(100, 401, 100), max_center_freq=random.randrange(4000, 7501, 1000), p=p_1)),
                        (5,HighShelfFilter(min_center_freq=random.randrange(100, 401, 100), max_center_freq=random.randrange(4000, 7501, 1000), p=p_1)),
                        (6,LowShelfFilter(min_center_freq=random.randrange(100, 401, 100), max_center_freq=random.randrange(4000, 7501, 1000), p=p_1)),
                        (7,PeakingFilter(min_center_freq=random.randrange(100, 401, 100), max_center_freq=random.randrange(4000, 7501, 1000), p=p_1))]
     # 随机选择一个噪声增强方法
    if augument_method:
        Filter_id, Filter_aug = Filter_type[augument_method]
    else:
        Filter_id, Filter_aug = random.choice(Filter_type)
    
    # 确保x是float32类型
    x = np.array(x, dtype=np.float32)
    
    # 应用噪声增强
    aug_filter_audio = Filter_aug(samples=x, sample_rate=16000)
    return aug_filter_audio, Filter_id
