# SCU

Code to accompany our 2018 IEEE International Conference on Systems, Man, and Cybernetics (SMC) paper entitled -
[On the classification of SSVEP-based dry-EEG signals via convolutional neural networks](https://arxiv.org/pdf/1805.04157.pdf).


## Usage

A working example demonstrating how to train a model can be found in the `Simple_train.py` file, which can be directly run with the included sample data in this repo. The base SCU model is included in the `SCU.py` file, and can be used to directly import the model into other codebases. 

## Dependencies and Requirements
The code has been designed to support python 3.6+ only. The project has the following dependencies and version requirements:

- torch=1.1.0+
- numpy=1.16++
- python=3.6.5+
- scipy=1.1.0+
- scikit-learn=0.23+

## Cite

Please cite the associated papers for this work if you use this code:

```
@inproceedings{aznan2018classification,
  title={On the classification of SSVEP-based dry-EEG signals via convolutional neural networks},
  author={Aznan, Nik Khadijah Nik and Bonner, Stephen and Connolly, Jason and Al Moubayed, Noura and Breckon, Toby},
  booktitle={2018 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  pages={3726--3731},
  year={2018},
  organization={IEEE}
}

```
