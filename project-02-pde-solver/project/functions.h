#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <Eigen/Dense>

// Function declarations
double analytic_solution(double x_Value, double y_Value, double t_value, double gamma);

double initial_cond(double x_Value, double y_Value);

double source_func(double x_value, double y_value, double t_value, double prev_value, double beta, double gamma, double alpha);

double source_func_linear(double x_value, double y_value, double t_value, double beta, double gamma, double alpha);

void write_to_file(const Eigen::MatrixXd& matrix, int timestep, int write_interval);

void compute_L2_norm(const Eigen::MatrixXd& numeric, const Eigen::MatrixXd& analytic, int timestep);


#endif // FUNCTIONS_H
