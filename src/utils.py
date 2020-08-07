def get_accuracy(actual, predicted):
    # actual: cuda longtensor variable
    # predicted: cuda longtensor variable
    assert(actual.size(0) == predicted.size(0))
    return float(actual.eq(predicted).sum()) / actual.size(0)