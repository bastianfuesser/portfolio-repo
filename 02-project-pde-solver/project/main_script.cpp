#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <math.h>
#include <cstdlib>
#include "functions.h"


// Variables
int gridpoints_x;
int gridpoints_y;
int timesteps;
int write_per_t;
double radi_coef_beta;
double freq_coef_gamma;
double coef_alpha;
double tolerance;



//Main function
int main(int argc, char* argv[]) {
    std::ofstream outputFile("output.log");
    if (!outputFile.is_open()) {
        std::cerr << "Unable to find log file for writing." << std::endl;
        return 1;
    }
    std::streambuf* originalOutput = std::cout.rdbuf();
    std::cout.rdbuf(outputFile.rdbuf());

    if (argc != 9) {
        std::cerr << "Usage: " << argv[0] << " <gridpoints_x> <gridpoints_y> <timesteps> <write_per_t> <radi_coef_beta> <freq_coef_gamma> <coef_alpha> <tolerance>" << std::endl;
        return 1;
    }

    gridpoints_x = std::atoi(argv[1]);
    gridpoints_y = std::atoi(argv[2]);
    timesteps = std::atoi(argv[3]);
    write_per_t = std::atoi(argv[4]);
    radi_coef_beta = std::atof(argv[5]);
    freq_coef_gamma = std::atof(argv[6]);
    coef_alpha = std::atof(argv[7]);
    tolerance = std::atof(argv[8]);

    std::cout << "gridpoints_x: " << gridpoints_x << std::endl;
    std::cout << "gridpoints_y: " << gridpoints_y << std::endl;
    std::cout << "timesteps: " << timesteps << std::endl;
    std::cout << "write_per_t: " << write_per_t << std::endl;
    std::cout << "radi_coef_beta: " << radi_coef_beta << std::endl;
    std::cout << "freq_coef_gamma: " << freq_coef_gamma << std::endl;
    std::cout << "coef_alpha: " << coef_alpha << std::endl;
    std::cout << "tolerance: " << tolerance << std::endl;

    double delta_x = 1.0 / gridpoints_x;
    double delta_y = 1.0 / gridpoints_y;
    double delta_t = 1.0 / timesteps;

    Eigen::MatrixXd matrixGrid(gridpoints_x, gridpoints_y);
    Eigen::MatrixXd updatedMatrix(gridpoints_x, gridpoints_y);


    // Initialize the Matrix-Grid at t=0. Diriclet boudary conditions are forced by if conditions.

    for (int i = 0; i < gridpoints_x; i++) {
        for (int j = 0; j < gridpoints_y; j++) {
            if (i == 0 || i == gridpoints_x - 1 || j == 0 || j == gridpoints_y - 1) {
                matrixGrid(i,j) = 0;
            } else {
                matrixGrid(i,j) = initial_cond(i * delta_x, j * delta_y);
            }
        }
    }

    // Initialize updated Matrix-Grid
    updatedMatrix = matrixGrid;
    write_to_file(updatedMatrix, 0, write_per_t);

    // If beta is zero, use the linear solving. In any other case use the Newton Rhapsody iteration method.
    if (radi_coef_beta == 0) {

        for (int t = 1; t < timesteps + 1; t++) {

            // Build linear matrix A
            Eigen::MatrixXd linear_matrix(gridpoints_x * gridpoints_y, gridpoints_x * gridpoints_y);
            linear_matrix.setZero();
            double temp_c = delta_t * coef_alpha / (delta_x * delta_x);
            for (int i = 0; i < gridpoints_x; i++) {
                for (int j = 0; j < gridpoints_y; j++) {
                    int temp_k = i * gridpoints_y + j;
                    linear_matrix(temp_k, temp_k) = 1 + 4 * temp_c;

                    if (j < gridpoints_y -1) {
                        linear_matrix(temp_k, temp_k + 1) = -temp_c;
                    }

                    if (j > 0) {
                        linear_matrix(temp_k, temp_k - 1) = -temp_c;
                    }

                    if (i < gridpoints_x -1) {
                        linear_matrix(temp_k, temp_k + gridpoints_x) = -temp_c;
                    }

                    if (i > 0) {
                        linear_matrix(temp_k, temp_k - gridpoints_x) = -temp_c;
                    }

                }
            }

            // Compute RHS of equation 9
            Eigen::VectorXd right_hand_side(gridpoints_x * gridpoints_y);
            for (int i = 0; i < gridpoints_x; i++) {
                for (int j = 0; j < gridpoints_y; j++) {
                    right_hand_side(i * gridpoints_y + j) = updatedMatrix(i, j) + delta_t * source_func_linear(i * delta_x, j * delta_y, t * delta_t, radi_coef_beta, freq_coef_gamma, coef_alpha);
                }
            }

            // Solve linear equation
            Eigen::VectorXd solved_equations = linear_matrix.colPivHouseholderQr().solve(right_hand_side);
            std::cout << "Bis hier vier" << std::endl;

            // Update matrix
            for (int i = 0; i < gridpoints_x; i++) {
                for (int j = 0; j < gridpoints_y; j++) {
                    if (i == 0 || i == gridpoints_x -1 || j == 0 || j == gridpoints_y -1) {
                        updatedMatrix(i, j) = 0;
                    } else {
                        updatedMatrix(i, j) = solved_equations(i * gridpoints_y + j);
                    }
                }
            }

            // Calculate matrix with analytic sulution
            if (t == timesteps) {
                matrixGrid.setZero();
                for (int i = 0; i < gridpoints_x; i++) {
                    for (int j = 0; j < gridpoints_y; j++) {
                        if (i == 0 || i == gridpoints_x -1 || j == 0 || j == gridpoints_y -1) {
                            matrixGrid(i, j) = 0;
                        } else {
                            matrixGrid(i, j) = analytic_solution(delta_x * i, delta_y * j, delta_t * t, freq_coef_gamma);
                        }
                    }
                }

                //Calcuate L2 norm
                compute_L2_norm(updatedMatrix, matrixGrid, t);
            }
            // Write to file
            write_to_file(updatedMatrix, t, write_per_t);


        }





    } else {
        int total_number_iterations = 0;
        for (int t = 1; t < timesteps + 1; t++) {
            Eigen::MatrixXd old_step = updatedMatrix;

             // Newton method
            for (int n = 0; n < 500; n++) {

                // Jacobian matrix
                Eigen::MatrixXd jacobian_matrix(gridpoints_x * gridpoints_y, gridpoints_x * gridpoints_y);
                jacobian_matrix.setZero();
                for (int i = 0; i < gridpoints_x; i++) {
                    for (int j = 0; j < gridpoints_y; j++) {
                        int temp_k = i * gridpoints_y + j;
                        double temp_c = delta_t * coef_alpha / (delta_x * delta_x);
                        jacobian_matrix(temp_k, temp_k) = 1 + 4 * temp_c + 4 * radi_coef_beta * delta_t * updatedMatrix(i, j) * updatedMatrix(i, j) * updatedMatrix(i, j);

                        if (j < gridpoints_x -1) {
                            jacobian_matrix(temp_k, temp_k + 1) = -temp_c;
                        }

                        if (j > 0) {
                            jacobian_matrix(temp_k, temp_k - 1) = -temp_c;
                        }

                        if (i < gridpoints_x -1) {
                            jacobian_matrix(temp_k, temp_k + gridpoints_x) = -temp_c;
                        }

                        if (i > 0) {
                            jacobian_matrix(temp_k, temp_k - gridpoints_x) = -temp_c;
                        }

                    }
                }
                int vec_lenght = gridpoints_x * gridpoints_y;
                Eigen::VectorXd g_equations(vec_lenght);

                for (int i = 0; i < gridpoints_x; i++) {
                    for (int j = 0; j < gridpoints_y; j++) {
                        if (i == 0 || j == 0 || i == gridpoints_x -1 || j == gridpoints_y -1) {
                            g_equations(i * gridpoints_y + j) = 0;
                        } else {
                            g_equations(i * gridpoints_y + j) = (
                                updatedMatrix(i, j)
                                - old_step(i, j)
                                - delta_t * coef_alpha * (updatedMatrix(i + 1, j) - 2 * updatedMatrix(i, j) + updatedMatrix(i - 1, j)) / (delta_x * delta_x)
                                - delta_t * coef_alpha * (updatedMatrix(i, j + 1) - 2 * updatedMatrix(i, j) + updatedMatrix(i, j - 1)) / (delta_y * delta_y)
                                - delta_t * source_func(i * delta_x, j * delta_y, t * delta_t, updatedMatrix(i, j), radi_coef_beta, freq_coef_gamma, coef_alpha)
                                );
                        }
                    }
                }

                // Solve linear equation
                Eigen::VectorXd solved_equations = jacobian_matrix.colPivHouseholderQr().solve(-g_equations);
 
                // Update used variables and enforce boundary conditions
                old_step = updatedMatrix;
                for (int i = 0; i < gridpoints_x; i++) {
                    for (int j = 0; j < gridpoints_y; j++) {
                        if (i == 0 || i == gridpoints_x -1 || j == 0 || j == gridpoints_y -1) {
                            updatedMatrix(i, j) = 0;
                        } else {
                            updatedMatrix(i, j) = solved_equations(i * gridpoints_y + j) + updatedMatrix(i, j);
                        }
                    }
                }
                
                std::cout << "The norm of iteration " << t << "-" << n <<" is " << solved_equations.norm() << std::endl ;

                // Terminate loop if precision is reached.
                if (solved_equations.norm() < tolerance) {
                    total_number_iterations += n ;
                    break;
                } else if (n == 99) {
                    total_number_iterations += n ;
                }
                
            }
        write_to_file(updatedMatrix, t, write_per_t);
        }
    std::cout << "The total amount of iterations was " << total_number_iterations << std::endl ;   
    }

    
    std::cout.rdbuf(originalOutput);
    std::cout << "Calculation sucessful!" << std::endl ;

    
    return 0;
}