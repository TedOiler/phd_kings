import numpy as np
import pandas as pd


def rand_design_matrix(experiments, features) -> pd.DataFrame:
    """
    :param int experiments: Number of Experiments to design
    :param int features: Number of features to include in the design


    Generate a random starting design matrix
    """
    df = pd.DataFrame(np.random.random((experiments, features)))
    df.columns = ['x' + str(x) for x in list(range(features))]
    return df


def get_model_matrix(data, order, interactions_only=False, bias=True) -> pd.DataFrame:
    """
    :param pd.DataFrame data: Design matrix
    :param int order: Order of the polynomial (1-main effects, 2-quadratic effects, ...)
    :param bool interactions_only: Include terms as x1^2 or not
    :param bool bias: Include a beta_0 on the design matrix or not


    Generate the model matrix of a design matrix (argument)
    """
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(order, interaction_only=interactions_only, include_bias=bias)
    df = pd.DataFrame(poly.fit_transform(data))
    df.columns = poly.get_feature_names(data.columns)
    return df


def cr_det(D, row, col, count, levels, model_order, interactions_only, bias, coordinate_cr) -> tuple:
    """
    :param pd.DataFrame D: Design matrix of the experiment
    :param int row: Row for coordinate to change in the design matrix
    :param int col: Col for coordinate to change in the design matrix
    :param int count:
    :param list levels: Levels of factors
    :param int model_order:
    :param bool interactions_only:
    :param bool bias:
    :param list coordinate_cr:


    """

    # run optimality criterion
    D.iat[row, col] = levels[count]  # change the design matrix to each level element
    model_matrix = get_model_matrix(data=D, order=model_order, interactions_only=interactions_only,
                                    bias=bias)  # calculate the design matrix
    det = np.linalg.det(model_matrix.T @ model_matrix)  # calculate the determinant
    coordinate_cr.append(det)  # write result on a list to find the max (and most importantly the max position)

    return D, coordinate_cr


def update_design_matrix(D, row, col, levels, coordinate_opt_cr) -> pd.DataFrame:
    """
    :param pd.DataFrame D: Design matrix of the experiment
    :param int row: Row for coordinate to change in the design matrix
    :param int col: Col for coordinate to change in the design matrix
    :param list levels: Levels of factors
    :param list coordinate_opt_cr: list of the optimality criterion for the specific coordinate

    Update the row (row) and the column (col) of the design matrix according to the criterion input by the user.
    """

    D.iat[row, col] = levels[coordinate_opt_cr.index(
        max(coordinate_opt_cr))]  # change the design matrix[row, col] element to the element of levels at the max position
    return D


def clear_histories(D, epochs, all_opt_cr, max_bool=True) -> tuple:
    """
    :param pd.DataFrame D: Number of Experiments to design
    :param int epochs: Number of random start to check
    :param list all_opt_cr: Levels of factors
    :param bool max_bool: Should the criterion be maximized (True) or minimizes (False)?

    Run the coordinate exchange algorithm and produce the best model matrix, according to the criterion chosen, as well as a history of all other possible model matrices and the history of the selected criterion used.
    """
    designs, histories = pd.DataFrame(), pd.DataFrame()
    D['epoch'] = epochs
    history = pd.DataFrame(all_opt_cr)
    if max_bool:
        history['max'] = history.iloc[:, 2:].max(axis=1)
    else:
        history['min'] = history.iloc[:, 2:].min(axis=1)

    history['epoch'] = epochs
    designs = designs.append(D, ignore_index=True)
    histories = histories.append(history, ignore_index=True)

    return designs, histories


def get_best_design(histories, designs, max_bool) -> pd.DataFrame:
    """
    :param pd.DataFrame histories: Dataframe of all the histories per epoch
    :param pd.DataFrame designs: Dataframe of all the designs per epoch
    :param bool max_bool: Should the criterion be maximized (True) or minimizes (False)?

    Group the histories per epoch and getting the max. Then, the function uses that max index (best epoch) to retrieve the design of that epoch and save it as the best design.
    The function also changes behaviour according to the max_bool flag which is used to tell the function if we are searching for a maximum of a minimum.
    """
    if max_bool:
        per_epoch = histories.groupby('epoch')['max'].max()
        best_design = designs[designs['epoch'] == per_epoch.idxmax()].reset_index().iloc[:, 1:-1]
    else:
        per_epoch = histories.groupby('epoch')['min'].min()
        best_design = designs[designs['epoch'] == per_epoch.idxmin()].reset_index().iloc[:, 1:-1]

    return best_design


def run_background_check() -> None:
    """

    If we have a second order model and two factors (-1 and 1) the design cannot be estimated. This is one example of syntactically correct but logically incorrect inputs.
    """
    pass


def coordinate_exchange(experiments, features, epochs, levels, model_order, interactions_only=False, bias=True,
                        criterion=cr_det, max_bool=True) -> tuple:
    """
    :param int experiments: Number of Experiments to design
    :param int features: Number of features to include in the design
    :param int epochs: Number of random start to check
    :param list levels: Levels of factors
    :param int model_order: Order of the polynomial (1-main effects, 2-quadratic effects, ...)
    :param bool interactions_only: Include terms as x1^2 or not
    :param bool bias: Include a beta_0 on the design matrix or not
    :param function criterion: What criterion to use for maximization. Includes (cr_det, ...)
    :param bool max_bool: Should the criterion be maximized (True) or minimizes (False)?

    Run the coordinate exchange algorithm and produce the best model matrix, according to the criterion chosen, as well as a history of all other possible model matrices and the history of the selected criterion used.
    """
    run_background_check()
    all_opt_cr = []  # used to feel the data for each level.

    for epoch in range(epochs):
        design_matrix = rand_design_matrix(experiments=experiments,
                                           features=features)  # create a random starting design for each epoch
        for row in range(experiments):
            for col in range(features):
                coordinate_opt_cr = []
                for count, level in enumerate(levels):
                    # run optimality criterion
                    design_matrix, coordinate_opt_cr = criterion(D=design_matrix,
                                                                 row=row, col=col, count=count,
                                                                 levels=levels,
                                                                 model_order=model_order,
                                                                 interactions_only=interactions_only,
                                                                 bias=bias,
                                                                 coordinate_cr=coordinate_opt_cr)

                # write results on loops
                all_opt_cr.append([row, col,
                                   *coordinate_opt_cr])  # keep entire history (unpack the coordinate_det to make all_dets a pd.DataFrame)
                design_matrix = update_design_matrix(D=design_matrix, row=row, col=col, levels=levels,
                                                     coordinate_opt_cr=coordinate_opt_cr)

        # clear results of loop
        designs, histories = clear_histories(D=design_matrix, epochs=epoch, all_opt_cr=all_opt_cr, max_bool=max_bool)

    # return best design
    best_design = get_best_design(histories=histories, designs=designs, max_bool=max_bool)

    # find best model matrix according to best design matrix
    model_matrix = get_model_matrix(data=best_design, order=model_order, interactions_only=False, bias=True)

    return model_matrix, best_design, designs, histories
