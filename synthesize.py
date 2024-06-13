import pickle
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pathlib import Path
import soundfile as sf
# import data_loader as loader

from voice_cloning.encoder import inference as encoder
from voice_cloning.encoder.params_model import model_embedding_size as speaker_embedding_size
from voice_cloning.synthesizer.inference import Synthesizer
from voice_cloning.utils.argutils import print_args
from voice_cloning.utils.default_models import ensure_default_models
from voice_cloning.vocoder import inference as vocoder

def synthesize_audio(synthesizer, vocoder, texts, embeds):
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    print(f'specs: {len(specs)}')
    breaks = [spec.shape[1] for spec in specs]
    spec = np.concatenate(specs, axis=1)
    print(f'spec: {spec.shape}')
    #print(specs[0].shape)
    #print("Created the mel spectrogram")

    ## Generating the waveform
        
    #print("Synthesizing the waveform:")

    # Synthesizing the waveform is fairly straightforward. Remember that the longer the
    # spectrogram, the more time-efficient the vocoder.
    
    generated_wav = vocoder.infer_waveform(spec)
    return generated_wav

def unique(list1):
    # initialize a null list
    unique_list = []
  
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return unique_list
