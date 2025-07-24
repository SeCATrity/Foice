# *Foice*: Can I Hear Your Face? Pervasive Attack on Voice Authentication Systems with a Single Face Image

This repository provides a PyTorch implementation of *Foice*.

*Foice* is a generative text-to-speech model that generates multiple synthetic audios from just a single image of the person’s face, without
requiring any voice sample.

Feel free to check out our demo video👉: https://drive.google.com/file/d/1Be1fgyDookg839UyV7DJbdBgx-YlD9ge/view?usp=sharing


## Dependencies
* face_alignment `pip install face-alignment`
* numpy
* cv2
* torch
* torchvision
* 

## Pre-trained models
| Face-dependent Voice Feature Extractor  | Face-independent Voice Feature Generator |
| --------------------------------------- | ---------------------------------------- |
| [link](https://drive.google.com/file/d/19H6uPPHkcRwOFza3dHx3xbln7-TOpCpH/view?usp=share_link) | [link](https://drive.google.com/file/d/1Ob7WZPtRGg5hT3plpoURcN-7bPlb69cn/view?usp=share_link)  |

## Voice generation

Foice reuses the `synthesizer` and `vocoder` from SV2TTS. You can find the pre-trained synthesizer using the [link](https://github.com/CorentinJ/Real-Time-Voice-Cloning).

Put all pre-trained models in the folder "../F2V_models/".

Run `End-to-End.ipynb` to generate voice recordings from image.

## Third-party related projects
* Image processing - face alignment: [face_alignment](https://github.com/1adrianb/face-alignment)
* Backbone text-to-speech model: [SV2TTS](https://github.com/CorentinJ/Real-Time-Voice-Cloning)

## TODO List
- [ ] Add pre-trained model
- [ ] Add training process
