import os
import random
from typing import Tuple
from scipy.io import loadmat
import torch
import glob

def parse_txt(config: dict) -> list:
    """Pases LibriSpeech metadata file to list of speakers with gender.

    Args:
        config (dict): Configuration conting the path to the metadata file of LibriSpeech.

    Returns:
        list: List conaining all speaker IDs and their corresponding gender.
    """
    file = config['paths']['meta-dir']
    fileHandle = open(file, 'r')
    id_gender = []
    for idx, line in enumerate(fileHandle):
        if idx > 11:
            fields = line.split('|')
            id_gender.append((fields[0].strip(), fields[1].strip()))
    fileHandle.close()
    return id_gender

def split_speakers(config: dict, ratio: float=0.5) -> Tuple[dict, dict]:
    """Function to split speakers between server training (train) and clustering algorithm (test).

    Args:
        config (dict): Configuration.
        ratio (float, optional): Split ratio of speakers. 1 is train only. Defaults to 0.5.

    Returns:
        Tuple[dict, dict]: Two dicts containing speakers with ID and gender that were split.
    """
    id_gender = parse_txt(config)  # get list of spk id and gender
    # Get the speaker ids
    speakers = {}
    for dirname, dirnames, filenames in os.walk(config['paths']['audio-dir']):
        for idx, dir_spk in enumerate(dirnames):
            speakers[idx] = {}
            speakers[idx]['id'] = dir_spk
            gen = [item[1] for item in id_gender if item[0] == dir_spk][0]
            speakers[idx]['gender'] = gen
        break

    len_train = int(ratio * len(speakers))
    speakers_test = dict(list(speakers.items())[len_train:])
    speakers_train = dict(list(speakers.items())[:len_train])

    return speakers_train, speakers_test

def get_source_speaker(config: dict, speakers: dict=None) -> dict:
    """Assigns speakers to sources.

    Args:
        config (dict): Configuration.
        speakers (dict, optional): Dict with available speakers containing gender and ID. Defaults to None.

    Returns:
        dict: Dict that assigns all sources to a speaker id with it's gender from the speaker dict.
    """
    random.seed(config['seeds']['random-seed'])

    id_gender = parse_txt(config)  # get list of spk id and gender

    matlab_data = loadmat(config['paths']['ir-meta-dir'])
    matlab_data["nsrc"][0][0]

    sources = []
    
    # Get all sources
    for i in range(matlab_data["nsrc"][0][0]):
        sources.append((chr(matlab_data["src_ids"][0][i]) + chr(matlab_data["src_ids"][1][i])))

    if speakers == None:
        speakers = {}
        # Get the speaker ids
        for dirname, dirnames, filenames in os.walk(config['paths']['audio-dir']):
            for idx, dir_spk in enumerate(dirnames):
                speakers[idx] = {}
                speakers[idx]['id'] = dir_spk
                gen = [item[1] for item in id_gender if item[0] == dir_spk][0]
                speakers[idx]['gender'] = gen
            break

    # Shuffle speakers list
    speakers_idx = list(speakers.keys())
    random.shuffle(speakers_idx)

    # Return src, speaker tuple
    combinations = {}
    tmp_gender = ''
    offset = 0
    for idx, src in enumerate(sources):
        if config['client']['gender_diff'] == True:
            while speakers[speakers_idx[idx+offset]]['gender'] == tmp_gender or len(glob.glob(os.path.join(config['paths']['audio-dir'], speakers[speakers_idx[idx+offset]]['id'])+ "/**/*.wav", recursive=True)) == 0:
                offset += 1
        elif config['client']['gender_same'] == True and idx != 0:
            while speakers[speakers_idx[idx+offset]]['gender'] != tmp_gender or len(glob.glob(os.path.join(config['paths']['audio-dir'], speakers[speakers_idx[idx+offset]]['id'])+ "/**/*.wav", recursive=True)) == 0:
                offset += 1
        
        combinations[idx] = {}
        combinations[idx]['source'] = src
        combinations[idx]['speaker'] = speakers[speakers_idx[idx+offset]]['id']
        combinations[idx]['gender'] = speakers[speakers_idx[idx+offset]]['gender']
        tmp_gender = speakers[speakers_idx[idx+offset]]['gender']

    return combinations


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor between -1 and 1.

    Subtract the mean, and scale to the interval [-1,1]

    Args:
        tensor (torch.Tensor): Input Tensor

    Returns:
        torch.Tensor: Output Tensor
    """
    tensor_minusmean = tensor - tensor.mean()
    return torch.div(tensor_minusmean, tensor_minusmean.abs().max())

def standardize(tensor):
    """Standardize tensor.

    Args:
        tensor (torch.Tensor): Input Tensor

    Returns:
        torch.Tensor: Output Tensor
    """
    means = tensor.mean(dim=0, keepdim=True)
    stds = tensor.std(dim=0, keepdim=True)
    return (tensor - means) / stds

def get_gender_dict() -> list:
    """Returns list of genders .

    Returns:
        dict: Dict ['M', 'F']
    """
    gender_dict = ['M', 'F']
    return gender_dict

def get_gender_index(gender: str) -> int:
    """Convert string of gender to ID of gender.

    Used for tranlating gender symbol to gender ID.

    Args:
        gender (str): Symbol of gender. ('M' or 'F')

    Returns:
        int: Index of gender (0 of 1)
    """
    gender_dict = ['M', 'F']

    return gender_dict.index(gender)

import numpy as np
import torchaudio
import torch.nn as nn
import logging

class FeatureHandler():
    def __init__(self, data: torch.utils.data.Dataset, device: str, config: dict, name: str, ir_data=None, transforms: torch.nn.Sequential=None):
        """Init of FeatureHandler.

        Used to unify feature computation across different devices.

        Args:
            data (torch.utils.data.Dataset): Dataset.
            device (str): Device name to use with PyTorch.
            config (dict): Configuration.
            name (str): Device name (e.g. client_1 or server)
            ir_data ([type], optional): [description]. Defaults to None.
            transforms (torch.nn.Sequential, optional): Additional feature transformation to apply. Defaults to None.
        """
        torchaudio.set_audio_backend('soundfile') # "sox", "sox_io" or "soundfile"
        self.ir_gain = 0.335294

        self.data = data # {idx: [audio], [speaker], [gender], [ir_idx], ['ir']}
        self.device = device
        self.config = config
        self.name = name
        self.ir = None
        self.ir_data = ir_data # {idx: room, source, receiver, file}

        self.logger = logging.getLogger(__name__)

        self.sr = self.config[config[name]['feature']]['sr']
        self.length =self.config[config[name]['feature']]['length']

        self.vad = None
        if self.config[self.config[self.name]['feature']]['vad'] == True:
            self.vad = nn.Sequential(
                    torchaudio.transforms.Vad(sample_rate=self.config[self.config[self.name]['feature']]['sr'])
                ).to(self.device)

        self.transforms = transforms

        if self.ir_data is not None:
            self.logger.debug("Loading IRs...")
            # self.get_gain()
            self.load_ir()
            self.logger.debug("[done]")

    def load_audio(self, file_path: str) -> torch.Tensor:
        """Load audio from file path.

        Used so every class uses same audio loading. This includes resampling, normalization, ...

        Args:
            file_path (str): Path to audio file.

        Returns:
            torch.Tensor: Tensor conting the processed audio file.
        """
        wav, _sr = torchaudio.load(file_path, normalization=True)
        # wav = wav.to(self.device) # To GPU
        wav = torch.mean(wav, 0) # Make mono
        
        # Resample
        resample = nn.Sequential(
                torchaudio.transforms.Resample(orig_freq=_sr, new_freq=self.sr)
            ).to(self.device)
        wav = resample(wav)

        if 'norm_utt' in self.config[self.name] and self.config[self.name]['norm_utt'] == 'normalize':
            wav = normalize(wav) # Normalize utterance
        elif 'norm_utt' in self.config[self.name] and self.config[self.name]['norm_utt'] == 'standardize':
            wav = standardize(wav)

        if self.config[self.config[self.name]['feature']]['vad'] == True:
            wav = self.vad(wav)

        return wav.to(self.device)
    
    def padding(self, wav: torch.Tensor) -> torch.Tensor:
        """Applies padding to the audio data.

        Args:
            wav (torch.Tensor): Input audio tensor.

        Returns:
            torch.Tensor: Padded audio tensor.
        """
        # Length correction
        if wav.shape[0] < self.length:
            diff = torch.tensor((self.length-wav.shape[0])/2, dtype=torch.int, device=self.device)
            pad = [int(diff), int(diff+1)]
            wav = torch.nn.functional.pad(wav, pad, mode='constant')
        
        return wav[:self.length]

    def convolve(self, wav: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Convoles two time singals using PyTorch functions.

        Uses conv1d from PyTorch.

        Args:
            wav (torch.Tensor): Utterance tensor.
            ir (torch.Tensor): IR tensor.

        Returns:
            torch.Tensor: Convolved tensor.
        """
        # Convolve
        a1 = wav.view(1, 1, -1).to(self.device) # Utterance with correct dimension for convolution
        b1 = torch.flip(ir, (0,)).view(1, 1, -1).to(self.device)  # IR with correct dimension for convolution
        wav = torch.nn.functional.conv1d(a1, b1, padding=len(ir,)-1).view(-1) # Convolution

        return wav

    def get_item(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Returns feature and label in a standardized way.

        Args:
            idx (int): Dataset index to compute feature for.

        Returns:
            Tuple[torch.Tensor, int]: Featrue and gender index as label.
        """
        audio, label = self.get_audio(idx)
        feature = self.get_feature(audio)

        return feature, label


    def get_audio(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Compute audio signal mixture of client for Dataset index.

        Convoles and normalizes audio data corresponding to configuration.
        It uses impulse response data for the signal computation.

        Args:
            idx (int): Database index to compute feature for.

        Returns:
            Tuple[torch.Tensor, int]: Resulting audio signal tensor, label as gender index.
        """
        # Index or impulse response
        ir_idx = None
        # Contains the combined audio signal in the end
        signal_combined = torch.zeros(self.length).to(self.device)

        # If the audio path is a list at the database index, perform a signal mix of all files.
        if isinstance(self.data[idx]['audio'], list):
            dominant = None # Dominatn sound source
            delay = None # shortest delay across all sources
            # Iterate over all audio files at index in database
            for spk_idx, audio in enumerate(self.data[idx]['audio']):
                file_path = self.data[idx]['audio'][spk_idx]
                # Load audio file
                wav = self.load_audio(file_path)

                # If impulse response index availoable in Dataset, load it for current speaker
                if 'ir_idx' in self.data[idx]:
                    ir_idx = self.data[idx]['ir_idx'][spk_idx]
                
                # If ir is available for this speaker
                if ir_idx is not None:
                    # Get IR data
                    ir = self.ir[ir_idx].to(self.device)

                    # Find shortest delay for determining dominant sound source of receiver
                    if spk_idx == 0 or torch.argmax(self.ir[ir_idx]) < delay:
                        delay = torch.argmax(self.ir[ir_idx])
                        dominant = spk_idx
                    
                    # Convolve speaker audio with impulse response
                    wav = self.convolve(wav, ir)

                # Padd audio signal if too short
                wav = self.padding(wav)
                
                # Add the new signal to the mix
                signal_combined = torch.add(signal_combined, wav)

            # Assign label as gender of dominant sound source
            label = get_gender_index(self.data[idx]['gender'][dominant])

        else:
            file_path = self.data[idx]['audio']

            if 'ir_idx' in self.data[idx]:
                ir_idx = self.data[idx]['ir_idx']

            # Load in audio
            wav = self.load_audio(file_path)

            if ir_idx is not None:
                ir = self.ir[ir_idx].to(self.device)

                # Convolve
                wav = self.convolve(wav, ir)

            signal_combined = self.padding(wav)

            label = get_gender_index(self.data[idx]['gender'])

        # Normalize
        if self.config[self.name]['normalize'] == True:
            signal_combined = normalize(signal_combined)
        elif False:
            signal_combined = torch.div(signal_combined, len(self.data[idx]['audio']))

        return signal_combined, label

    def get_feature(self, wav: torch.Tensor) -> torch.Tensor:
        """Computes features standardized for all devices.

        Args:
            wav (torch.Tensor): Input audio tensor.

        Returns:
            torch.Tensor: Output feature tensor. (e.g. lmbe)
        """
            
        # If no transformation is given, use this one
        if self.transforms is None:
            n_fft = self.config[self.config[self.name]['feature']]['n_fft']
            hop_length = self.config[self.config[self.name]['feature']]['hop_length']
            n_mels = self.config[self.config[self.name]['feature']]['n_mels']
            f_min = self.config[self.config[self.name]['feature']]['fmin']
            f_max = self.config[self.config[self.name]['feature']]['fmax'] // 2
            top_db = self.config[self.config[self.name]['feature']]['top_db']

            transforms = nn.Sequential(
                torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, f_min=f_min, f_max=f_max),
                torchaudio.transforms.AmplitudeToDB(stype='power', top_db=top_db),
            ).to(self.device)
        else:
            transforms = self.transforms
        
        # Apply transformation
        spec = transforms(wav)

        # Normalize
        spec_norm = torch.nn.functional.normalize(spec)
        spec_norm -= spec_norm.min(1, keepdim=True)[0]

        # Shape to [1, sec, x, x]
        size = np.shape(spec_norm)
        if 'no_change' in self.config[self.config[self.name]['feature']]:
            spec_norm = torch.transpose(spec_norm, 0, 1)
        elif 'length_lmbe' in self.config[self.config[self.name]['feature']]:
            length = self.config[self.config[self.name]['feature']]['length_lmbe']
            limit = np.int(size[1]/length)*length
            spec_norm = spec_norm[:, 0:limit]
            spec_norm = torch.reshape(spec_norm, (np.int(size[1] / length), length, size[0]))
        else:
            limit = np.int(size[1]/size[0])*size[0]
            spec_norm = spec_norm[:, 0:limit]
            spec_norm = torch.reshape(spec_norm, (np.int(size[1] / size[0]), size[0], size[0]))

        return spec_norm[np.newaxis, ...]

    def get_gain(self) -> torch.Tensor:
        """Get gain suitable for all IRs in the database

        Reads in all impulse responses that can be found in the ir-dir folder and gets the highest amplitude.
        The gain is the inverse of the maximum amplitide.

        Returns:
            torch.Tensor: Gain to apply to IRs
        """
        # Get IR gain
        max_amp = -1
        for subdir, dirs, files in os.walk(self.config['paths']['ir-dir']):
            for file in files:
                if file.lower().endswith(".wav"):
                    ir_path = os.path.join(subdir, file)
                    ir, _sr = torchaudio.load(ir_path, normalization=True) # To GPU
                    resample = nn.Sequential(
                            torchaudio.transforms.Resample(orig_freq=_sr, new_freq=self.config[self.config[self.name]['feature']]['sr'])
                        )
                    ir = resample(ir)
                    tmp_max = torch.max(torch.abs(ir))
                    if max_amp < tmp_max:
                        max_amp = tmp_max
        self.ir_gain = 1/max_amp

        self.logger.info('New gain: %f' % (self.ir_gain))

        return self.ir_gain

    def load_ir(self) -> list:
        """Preloads all impulse responses of client.

        Returns:
            list of torch.Tensor: List of all impulse respones corresponding to client/node.
        """
        self.ir = []
        for ir_idx in self.ir_data:
            # Load in IR
            ir, _sr = torchaudio.load(self.ir_data[ir_idx]['file'], normalization=True)
            # ir = ir.to(self.device) # To GPU

            # Resample
            resample = nn.Sequential(
                    torchaudio.transforms.Resample(orig_freq=_sr, new_freq=self.config[self.config[self.name]['feature']]['sr'])
                ).to(self.device)
            ir = resample(ir)

            # Apply gain
            ir = self.ir_gain * torch.mean(ir, 0) # gain + mono
            ir = ir.to(self.device) # To GPU
            self.ir.append(ir)

        return self.ir