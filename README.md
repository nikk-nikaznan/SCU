# SCU

Code to accompany our 2018 IEEE International Conference on Systems, Man, and Cybernetics (SMC) paper entitled -
[On the classification of SSVEP-based dry-EEG signals via convolutional neural networks](https://arxiv.org/pdf/1805.04157.pdf).


## Usage

To run the code using the sample data, run
```
python src/scu/EEG_1SCU_CM.py
```
SCU model is included in the `model.py`, and can be used to directly import the model into other codebases. 

## Requirements
To install the dependencies using `conda`, run

```
  conda create -n scu -c conda-forge python=3.10
  conda activate scu
  pip install -e ".[dev]"
```

Alternatively, to install the dependencies using `pyenv`, run

```
  ~/.pyenv/versions/3.10.12/bin/python -m venv ~/Venvs/scu
  source ~/Venvs/scu/bin/activate
  pip install -e ".[dev]"
```

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
