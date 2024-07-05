class LinearRegression:
    def __init__(self, n_features, n_output):
        """
        params :
            n_features [INT] : The number of columns in the input data
            n_output [INT] : The number of columns in the output data
        """
        self.weight = [[0 for cols in range(n_features)] for rows in range(n_output)]
        self.bias = [0 for cols in range(n_output)]
        # Hyperparameters of the model
        self.tol = 1e-6
        self.n_iters = 200
        self.learning_rate = 1e-3

    def forward(self, x):
        """
        Params :
            x [List[List]]: The matrix of input data containing independent features

        Return:
            The predictions are returned for the input data.
        """
        return self.matrix_add(self.matrix_multiply(x, self.transpose(self.weight)), [self.bias for row in range(len(x))])

    def loss_function(self, y_actual, y_predicted):
        """
        Params :
            y_actual [List[List]] : The actual labels
            y_predicted [List[List]] : The predicted labels

        Returns : The Mean Squared Error metric is computed and returned.
        """
        pointwise_loss = self.matrix_add(y_actual, y_predicted, is_subtraction = True, apply_square = True)
        total_loss = 0
        for row in range(len(pointwise_loss)):
            for col in range(len(pointwise_loss[row])):
                total_loss += pointwise_loss[row][col]
        return total_loss
    
    def fit(self, x, y):
        """
        Params : 
            x [List[List]] : The input data containing independent featurs.
            y [List[List]] : The output data containing dependent features for training the model.

        Returns : The convergence status of the model.
        """
        prediction = self.forward(x)
        loss = self.loss_function(y, prediction)
        status = "not converged"
        if loss < self.tol:
            status = "converged"
        else:
            iters = 0
            while(status == "not converged"):
                prediction = self.forward(x)
                loss = self.loss_function(y, prediction)
                # Gradient of Loss Function with respect to the weights and biases are calculated.
                dl_dw = self.matrix_multiply(self.transpose(x), self.matrix_add(y, prediction, is_subtraction = True, apply_square = False))
                dl_db_ = self.matrix_add(y, prediction, is_subtraction = True, apply_square = False)
                dl_db = []
                for col in range(len(dl_db_[0])):
                    partial_sum = 0
                    for row in range(len(dl_db_)):
                        partial_sum += dl_db_[row][col]
                    dl_db.append(self.learning_rate * partial_sum)

                for row in range(len(dl_dw)):
                    for col in range(len(dl_dw[row])):
                        dl_dw[row][col] = self.learning_rate * dl_dw[row][col]
                # Weights and Biases are updated by subtracting the repective gradients.
                self.weight = self.matrix_add(self.weight, self.transpose(dl_dw), is_subtraction = True, apply_square = False)
                self.bias = self.matrix_add([self.bias], [dl_db], is_subtraction = True, apply_square = False)[0]
                iters += 1
                if loss < self.tol:
                    status = "converged"
                if iters >= self.n_iters:
                    # status = "converged"
                    break
        return status

    def transpose(self, matrix):
        """
        Params :
            matrix [List[List]] : The matrix that needs to be transposed.

        Returns : The transposed matrix.
        """
        transposed_matrix = [[0 for col in range(len(matrix))] for row in range(len(matrix[0]))]
        for row in range(len(transposed_matrix)):
            for col in range(len(transposed_matrix[row])):
                transposed_matrix[row][col] = matrix[col][row]
        return transposed_matrix
    
    def matrix_multiply(self, matrix1, matrix2):
        """
        Params : 
            matrix1 [List[List]] : The Left matrix to be multiplied
            matrix2 [List[List]] : The Right matrix to be multiplied

        Returns : The product of two matrices is returned.
        """
        assert len(matrix1[0]) == len(matrix2), f"The matrix input arguments are not multiply compatible. Left matrix is of shape {len(matrix1)} X {len(matrix1[0])} and Right matrix is of shape {len(matrix2)} X {len(matrix2[0])}"
        rows = []
        for row in range(len(matrix1)):
            cols = []
            for col in range(len(matrix2[0])):
                partial_sum = 0
                for k in range(len(matrix2)):
                    partial_sum += matrix1[row][k] * matrix2[k][col]
                cols.append(partial_sum)
            rows.append(cols)
        return rows

    def matrix_add(self, matrix1, matrix2, is_subtraction = False, apply_square = False):
        """
        Params : 
            matrix1 [List[List]] : The Left matrix to be added
            matrix2 [List[List]] : The Right matrix to be added
            is_subtraction [Bool] : The flag expressing that instead of addition, subtraction of two matrices needs to be performed.
            apply_square [Bool] : The flag representing that element wise squared difference needs to be computed.

        Returns : The result of elementwise addition / subtraction / squared difference is returned based on the choice of the flag.
        """
        assert len(matrix1) == len(matrix2) and len(matrix1[0]) == len(matrix2[0]), f"The matrix input arguments are not addition compatible. Left matrix is of shape {len(matrix1)} X {len(matrix1[0])} and Right matrix is of shape {len(matrix2)} X {len(matrix2[0])}" 
        for row in range(len(matrix1)):
            for col in range(len(matrix1[row])):
                if is_subtraction == False:
                    matrix1[row][col] = matrix1[row][col] + matrix2[row][col]
                else:
                    if apply_square == False:
                        matrix1[row][col] = matrix1[row][col] - matrix2[row][col]
                    else:
                        # Modulo Division is added to handle the overflow error.
                        matrix1[row][col] = ((matrix1[row][col] - matrix2[row][col]) ** 2) % 1e-6
        return matrix1
    


if __name__ == "__main__":
    import random
    n_features = 10
    n_output = 1
    x = [[random.random() for col in range(10)] for row in range(10000)]
    y = [[random.random() for col in range(1)] for row in range(10000)]
    model = LinearRegression(n_features, n_output)

    prediction = model.forward(x)
    print("Loss (Before Fitting to data) : ", model.loss_function(y, prediction))

    model.fit(x, y)
    prediction = model.forward(x)
    print("Loss (After Fitting to data) : ", model.loss_function(y, prediction))

