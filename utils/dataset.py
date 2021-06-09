import os
import numpy as np
from scipy.io import loadmat
import random
import logging
from scipy.spatial import distance
import itertools

import torch
from torch._C import device, dtype
import torchaudio
import torchaudio.transforms
import torch.nn as nn
from torch.utils.data import Dataset

from utils.data_helper import *
from utils.util import *

class LibriData(Dataset):
    def __init__(self, transforms: torch.nn.Sequential=None, device: str=None, config: dict=None, combinations:dict=None, name: str="server", speakers: dict=None):
        """Init of LibriData.

        Dataset for using only clean audio from the LibriSpeech Dataset.

        Args:
            transforms (torch.nn.Sequential, optional): Audio feature extraction overwrite. Defaults to None.
            device (str, optional): PyTorch device to train on. Defaults to None.
            config (dict, optional): Configuration. Defaults to None.
            combinations (dict, optional): Source speaker combination dict. Defaults to None.
            name (str, optional): Name of device. Defaults to "server".
            speakers (dict, optional): Dict of speakers with id and gender. Defaults to None.
        """
        self.data = {}
        self.combinations = combinations
        self.name = name
        self.speakers = speakers
        self.logger = logging.getLogger(__name__)
        
        self.config = config
        self.transforms = transforms

        if device == None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        random.seed(config['seeds']['random-seed'])
        # Create self.data
        self.prepare_database(config)

        # Instance of FeatureHandler for unified feature computation
        self.feature_handler = FeatureHandler(self.data, self.device, self.config, self.name, transforms=self.transforms)
            
    def __len__(self) -> int:
        """Returns length of dataset.

        Returns:
            int: Length of dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Returns feature with label.

        Args:
            idx (int): Index of data in dataset.

        Returns:
            Tuple[torch.Tensor, int]: Computed feature and gender index
        """
        feature, label = self.feature_handler.get_item(idx)
        
        return feature, label

    def get_sample_data(self, idx):
        """Returns a sample of data.

        Used to get both, audio and feature for data with corresponding gender

        Args:
            idx (int): Index of data in the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, list]: Audio tensor, feature tensor, gender of data as symbol and index
        """
        audio, label = self.feature_handler.get_audio(idx)
        feature = self.feature_handler.get_feature(wav=audio)

        meta = [self.data[idx]['gender'], label]

        return audio, feature, meta

    def prepare_database(self, config: dict):
        """Prepares dataset by creating data dict.

        Gets all audio files from LibriSpeech with their path, speaker id, and gender.

        Args:
            config (dict): Configuration
        """
        # Get list of speaker speaker and gender
        id_gender = parse_txt(config)

        idx = 0
        # Get all utterance audio files and save with speaker in dict
        for dirname, dirnames, filenames in os.walk(config['paths']['audio-dir']):
            for i, filename in enumerate(filenames):
                if filename.endswith('.txt'):
                    continue
                pos = filename.find('-')
                speaker = filename[0:pos]
                pos2 = filename[pos+1:-1].find('-')
                track = filename[pos+1:pos+pos2+1]
                audio = os.path.join(config['paths']['audio-dir'], speaker, track, filename)
                if (
                    self.config[self.name]['use_all'] == True or 
                    (self.speakers == None and self.combinations == None) or
                    (
                        speaker in [entry['id'] for entry in self.speakers.values()]
                        # and speaker not in [combination['speaker'] for combination in self.combinations.values()]
                    )
                    ):
                    
                    self.data[idx] = {}
                    self.data[idx]['audio'] = audio
                    self.data[idx]['speaker'] = int(speaker)
                    gen = [item[1] for item in id_gender if item[0] == speaker][0]
                    self.data[idx]['gender'] = gen
                    idx += 1
        
        self.logger.info("Dataset uses %d files" % (len(self.data)))

class NodeData(Dataset):
    def __init__(self, rec: int, combinations: dict, transforms: torch.nn.Sequential=None, device: str=None, config: dict=None, name: str="client"):
        """Init of NodeData Dataset.

        Creates data for nodes. This includes loading the correct IRs to the sources receiver combinations.

        Args:
            rec (int): Node/Client ID
            combinations (dict): Combination of sources and speakers
            transforms (torch.nn.Sequential, optional): Feature computation overwrite. Defaults to None.
            device (str, optional): PyTorch device to train on. Defaults to None.
            config (dict, optional): Configuration. Defaults to None.
            name (str, optional): Name of device (e.g. client_1). Defaults to "client".
        """
        self.speaker_data = {} # Dict with speaker data
        self.sources = [] # List of sources
        self.min_length = 0 # Minimum number of songs through speakers
        self.ir_data = {}
        
        self.rec = rec # Receiver corresponding to this node
        self.dominant = None # Dominant sound source
        self.ground_truth = None # Ground truth value
        self.combinations = combinations # Combinations of sources and speakers
        self.config = config # General config
        self.name = name

        self.logger = logging.getLogger(__name__)

        self.transforms = transforms

        if device == None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        random.seed(config['seeds']['random-seed'])

        # Prepare database
        self.prepare_database(config)

        self.feature_handler = FeatureHandler(self.data, self.device, self.config, self.name, ir_data=self.ir_data, transforms=self.transforms)
            
    def __len__(self):
        """Returns length of dataset.

        Returns:
            int: Length of dataset
        """
        return self.min_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Returns feature with label.

        Args:
            idx (int): Index of data in dataset.

        Returns:
            Tuple[torch.Tensor, int]: Computed feature and gender index
        """
        feature, label = self.feature_handler.get_item(idx)
        
        return feature, label

    def get_sample_data(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """Returns a sample of data.

        Used to get both, audio and feature for data with corresponding gender

        Args:
            idx (int): Index of data in the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, list]: Audio tensor, feature tensor, gender of data as symbol and index
        """
        audio, label = self.feature_handler.get_audio(idx)
        feature = self.feature_handler.get_feature(wav=audio)

        meta = [self.data[idx]['gender'], label]

        return audio, feature, meta

    def prepare_database(self, config: dict):
        """Prepare database for client/node

        Args:
            config (dict): Configuration dict.
        """
        from utils.util import critical_distance
        # Get list of speaker speaker and gender
        id_gender = parse_txt(config)

        # Get MATLAB data from CATT-A with information about RIRs
        self.matlab_data = loadmat(config['paths']['ir-meta-dir'])

        # Add paths if not in config
        if "ir-dir" not in config['paths']:
            self.config['paths']['ir-dir'] = os.path.dirname(self.config['paths']['ir-meta-dir'])
        if "room-dir" not in config['paths']:
            self.config['paths']['room-dir'] = os.path.join(os.path.dirname(self.config['paths']['ir-dir']), 'IN')
        if "room" not in config['paths']:
            self.config['paths']['room'] = '_'.join([self.config['paths']['ir-dir'].split('_')[0], self.config['paths']['ir-dir'].split('_')[1]])

        # Get all sources
        for i in range(self.matlab_data["nsrc"][0][0]):
            self.sources.append((chr(self.matlab_data["src_ids"][0][i]) + chr(self.matlab_data["src_ids"][1][i])))

        # Get position of client
        self.position = [self.matlab_data["rec_positions"][i][self.rec] for i in range(3)]

        # Get ground truth cluster based on critical distance
        for i in range(self.matlab_data["nsrc"][0][0]):
            t30 = np.mean(self.matlab_data['T30_E'][:][0:-2]) # Reverberation time
            r_H = critical_distance(self.matlab_data["room_volume"][0][0], t30) # Critical distance
            # Source position
            src_pos = [self.matlab_data["src_positions"][0][i], self.matlab_data["src_positions"][1][i], self.matlab_data["src_positions"][2][i]]
            # If inside critical distance
            if distance.euclidean(src_pos, self.position) < r_H:
                self.ground_truth = i # ID of source
        if self.ground_truth == None:
            self.ground_truth = self.matlab_data["nsrc"][0][0]
        # If not in any critical distance -> Background cluster
        if self.ground_truth >= len(self.combinations):
            cluster_name = "B"
        else:
            cluster_name = self.combinations[self.ground_truth]['source'] # Name of source
        self.logger.debug("Ground truth cluster: %d (%s)" % (self.ground_truth, cluster_name))

        # Get simulation count from CATT-A for loading the right RIRs
        project_name = ''.join([chr(char) for char in self.matlab_data["project"][0]])
        cnt = int(self.config['paths']['room'].split('_')[-1])

        # Get the speaker IDs with gender and init empty utterance list
        for root, dirnames, filenames in os.walk(config['paths']['audio-dir']):
            for idx, dir_spk in enumerate(dirnames):
                self.speaker_data[idx] = {}
                self.speaker_data[idx]['speaker'] = dir_spk
                gen = [item[1] for item in id_gender if item[0] == dir_spk][0]
                self.speaker_data[idx]['gender'] = gen
                self.speaker_data[idx]['utt'] = []
                self.speaker_data[idx]['utt_count'] = 0
            break

        # Get audio paths for all utterance files and save in list corresponding to speaker
        for root, dirnames, filenames in os.walk(config['paths']['audio-dir']):
            for filename in filenames:
                if filename.endswith('.txt'):
                    continue
                pos = filename.find('-')
                speaker = filename[0:pos] # Speaker ID
                pos2 = filename[pos+1:-1].find('-')
                track = filename[pos+1:pos+pos2+1] # Sub-folder of speaker
                audio = os.path.join(config['paths']['audio-dir'], speaker, track, filename)

                # Add all utterance file to right speaker
                for key, value in self.speaker_data.items():
                    if value['speaker'] == speaker:
                        self.speaker_data[key]['utt'].append(audio)
                        self.speaker_data[key]['utt_count'] += 1

        # Remove speakers not used by source
        speakers = [] # Array of speakers to keep
        new_speaker_data = {} # Array of used speakers
        comb_idx = 0 # iterate over combination array
        for key, value in self.combinations.items():
            speakers.append(value['speaker'])
        for key, value in self.speaker_data.items():
            if value['speaker'] in speakers:
                self.speaker_data[key]['source'] = self.combinations[comb_idx]['source'] # Save corresponding source
                # Save RIR path for corresponding source and receiver of this node
                source_file = "%s_%d_%s_%02d_%s.WAV" % (project_name, cnt, self.combinations[comb_idx]['source'], self.matlab_data['rec_ids'][0][self.rec], 'OMNI') # Get right file name
                self.speaker_data[key]['source_file'] = os.path.join(config['paths']['ir-dir'], source_file) # Combine with path
                new_speaker_data[comb_idx] = self.speaker_data[key]
                comb_idx += 1
        self.speaker_data = new_speaker_data

        # Get minimum count of utterance files among all speakers
        self.min_length = self.speaker_data[0]['utt_count']
        for key, value in self.speaker_data.items():
            if self.min_length > value['utt_count']:
                self.min_length = value['utt_count']

        self.logger.debug("Min utterances count: %d" % (self.min_length))

        # Reduce number of utterances if set in config
        if self.config['client']['max_data'] != -1 and self.min_length > self.config['client']['max_data']*(1+self.config['client']['train_frac']):
            self.min_length = int(np.ceil(self.config['client']['max_data']*(1+self.config['client']['train_frac'])))
            self.logger.debug("Reduced to: %d" % (self.min_length))

        # Load RIRs and get dominant sound source for client
        delay = 0
        max_a = 0
        for src_id, src in enumerate(self.sources):
            ir, _sr = torchaudio.load(self.speaker_data[src_id]['source_file'], normalization=True)
            if src_id == 0 or torch.argmax(ir) < delay:
                delay = torch.argmax(ir)
                max_a = torch.max(ir)
                self.dominant = src_id

            path = self.speaker_data[src_id]['source_file']
            file = os.path.basename(path)
            subdir = os.path.dirname(path)
            self.ir_data[src_id] = {}
            self.ir_data[src_id]['room'] = '_'.join([file.split('_')[0], file.split('_')[1]])
            self.ir_data[src_id]['source'] = file.split('_')[2]
            self.ir_data[src_id]['receiver'] = file.split('_')[3]
            self.ir_data[src_id]['file'] = os.path.join(subdir, file)

            self.logger.debug("Receiver %d Source %s: %d -> %f" % (self.rec, src, torch.argmax(ir), torch.max(ir)))

        self.logger.info("Dominant source of receiver %d is %s" % (self.rec, self.sources[self.dominant]))

        # Get data with speaker and corresponding irs
        self.data = {}
        for utt_idx in range(self.min_length):
            self.data[utt_idx] = {}
            self.data[utt_idx]['audio'] = []
            self.data[utt_idx]['speaker'] = []
            self.data[utt_idx]['gender'] = []
            self.data[utt_idx]['ir_idx'] = []
            self.data[utt_idx]['ir'] = []
            for key, value in self.speaker_data.items():
                self.data[utt_idx]['audio'].append(value['utt'][utt_idx])
                self.data[utt_idx]['speaker'].append(value['speaker'])
                self.data[utt_idx]['gender'].append(value['gender'])
                self.data[utt_idx]['ir_idx'].append(key)
                self.data[utt_idx]['ir'].append(self.speaker_data[key]['source_file'])

        self.logger.debug("Dataset uses %d files" % (len(self.data)))