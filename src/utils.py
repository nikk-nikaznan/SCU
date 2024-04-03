import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import itertools
from typing import List


def load_data() -> np.ndarray:
    """
    Load EEG data from file.

    Returns:
        np.ndarray: EEG data.
    """
    # data loading
    EEGdata = np.load("data/SampleData_S01_4class.npy")
    input_data = EEGdata.swapaxes(1, 2)
    return input_data


def load_label() -> np.ndarray:
    """
    Load labels from file.

    Returns:
        np.ndarray: Label data.
    """
    # data loading
    input_label = np.load("data/SampleData_S01_4class_labels.npy")
    return input_label


def plot_error_matrix(
    cm: np.ndarray, classes: List[str], cmap: plt.cm.Blues = plt.cm.Blues
) -> None:
    """
    Plot the error matrix for the neural network models.

    Args:
        cm (np.ndarray): Confusion matrix.
        classes (List[str]): List of class names.
        cmap (plt.cm.Blues, optional): Color map for the plot. Defaults to plt.cm.Blues.

    Returns:
        None
    """
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


def CCM(cnf_labels: np.ndarray, cnf_predictions: np.ndarray) -> None:
    """
    Generate and plot the confusion matrix.

    Args:
        cnf_labels (np.ndarray): True labels.
        cnf_predictions (np.ndarray): Predicted labels.

    Returns:
        None
    """
    class_names = ["10", "12", "15", "30"]
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(cnf_labels, cnf_predictions)
    np.set_printoptions(precision=2)

    # Normalize
    cnf_matrix = cnf_matrix.astype("float") / cnf_matrix.sum(axis=1)[:, np.newaxis]
    cnf_matrix = cnf_matrix.round(4)
    matplotlib.rcParams.update({"font.size": 16})

    # Plot normalized confusion matrix
    plt.figure()
    plot_error_matrix(cnf_matrix, classes=class_names)
    plt.tight_layout()
    filename = "S01_SCU.pdf"
    plt.savefig(filename, format="PDF", bbox_inches="tight")
    # plt.show()
