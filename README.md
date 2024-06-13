# *Foice*: Can I Hear Your Face? Pervasive Attack on Voice Authentication Systems with a Single Face Image

This repository provides a PyTorch implementation of *Foice*.

*Foice* is a generative text-to-speech model that generates multiple synthetic audios from just a single image of the person’s face, without
requiring any voice sample.



## Dependencies
* face_alignment `pip install face-alignment`

## Pre-trained models
| Face-dependent Voice Feature Extractor  | Face-independent Voice Feature Generator |
| --------------------------------------- | ---------------------------------------- |
| [link](https://drive.google.com/file/d/19H6uPPHkcRwOFza3dHx3xbln7-TOpCpH/view?usp=share_link) | [link](https://drive.google.com/file/d/1Ob7WZPtRGg5hT3plpoURcN-7bPlb69cn/view?usp=share_link)  |

## Voice generation

Foice reuses the synthesizer from SV2TTS. You can find the pre-trained synthesizer using the [link](https://github.com/CorentinJ/Real-Time-Voice-Cloning).

## Third-party related projects
* Image processing - face alignment: [face_alignment](https://github.com/1adrianb/face-alignment)
* Backbone text-to-speech model: [SV2TTS](https://github.com/CorentinJ/Real-Time-Voice-Cloning)

## TODO List
- [ ] Add pre-trained model
- [ ] Add training process
