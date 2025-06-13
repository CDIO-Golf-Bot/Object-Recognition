def significant_change(balls, last, tol_cm=1.0):
    """
    Returns True if the new ball list differs significantly from the last.
    """
    if len(balls) != len(last):
        return True
    for (x, y, label, *_), (lx, ly, llabel, *_ ) in zip(balls, last):
        if label != llabel or abs(x - lx) > tol_cm or abs(y - ly) > tol_cm:
            return True
    return False
