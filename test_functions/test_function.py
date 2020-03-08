class TestFunction:
    r"""Super class to implement a test function.

    Arguments:
        x_domain: iterable of int for the x search domain
        y_domain: iterable of int for the y search domain
        minimum: global minimum (x0, y0) of the
            function for the given domain
        initial_state: starting point
        num_pt: number of point in the search domain
        levels: determines the number and positions of the contour lines / regions.
    """

    def __init__(
        self,
        x_domain: iter,
        y_domain: iter,
        minimum: iter,
        initial_state: iter,
        num_pt: int = 250,
        levels: int = 50,
    ):
        self._x_domain = x_domain
        self._y_domain = y_domain
        self._minimum = minimum
        self._initial_state = initial_state
        self._num_pt = num_pt
        self._levels = levels

    def __call__(self, tensor, lib):
        raise NotImplementedError

    def __name__(self):
        return self.__class__.__name__

    @property
    def x_domain(self):
        return self._x_domain

    @property
    def y_domain(self):
        return self._y_domain

    @property
    def num_pt(self):
        return self._num_pt

    @property
    def minimum(self):
        return self._minimum

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def levels(self):
        return self._levels
