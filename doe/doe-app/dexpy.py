import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures  # TODO: implement what I need from this package


class DesignParams:
    pass


class Design:
    """
    Class Docstring.
    """
    def __init__(self, experiments=None, levels=None):
        """
        :param int experiments: Number of Experiments to design
        :param dict levels: Levels of factors
        Constructor Docstring.
        """

        self.experiments = experiments
        self.levels = levels
        self.features = len(levels.keys())

        self.order = None
        self.interactions_only = None
        self.bias = None

        self.epochs = None
        self.criterion = None

# ---------------DUNDER, GETTERS AND SETTERS FUNCTION-------------------------------------------------------------------
    def __repr__(self): return f"Design(experiments={self.experiments}, levels={self.levels})"

    def set_model(self, order, interactions_only=False, bias=True):
        """
        :param int order: Order of the polynomial (1-main effects, 2-quadratic effects, ...)
        :param bool interactions_only: Include terms as x1^2 or not
        :param bool bias: Include a beta_0 on the design matrix or not

        Setter for model parameters
        """
        self.order = order
        self.interactions_only = interactions_only
        self.bias = bias

    def set_algorithm(self, epochs, criterion):
        """
        :param int epochs: Number of random start to check
        :param str criterion: What criterion to use for maximization. Includes ("A", "C", "D", "E", "S", "T", "G", "I", "V")

        Setter for algorithm parameters
        """
        self.epochs = epochs
        self.criterion = criterion
# ----------------------------------------------------------------------------------------------------------------------

    def gen_random_design(self) -> pd.DataFrame:
        """
        Generate a random starting design matrix.
        """
        df = pd.DataFrame(np.random.random((self.experiments, self.features)))
        df.columns = ['x' + str(x) for x in list(range(self.features))]
        return df

    def gen_model_matrix(self, data=None) -> pd.DataFrame:
        """
        :param pd.DataFrame data: Design matrix

        Generate the model matrix of a design matrix (argument)
        """

        poly = PolynomialFeatures(degree=self.order,
                                  interaction_only=self.interactions_only,
                                  include_bias=self.bias)
        df = pd.DataFrame(poly.fit_transform(data))
        df.columns = poly.get_feature_names(data.columns)
        return df

    def fit(self):

        pass
