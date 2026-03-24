#include "functions.h"
#include <iostream>
#include <fstream>
#include <vtkSmartPointer.h>
#include <vtkStructuredPoints.h>
#include <vtkStructuredPointsWriter.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <math.h>
#include <cstdlib>

double analytic_solution(double x_Value, double y_Value, double t_value, double gamma) {
    return( std::sin(M_PI * x_Value) * std::sin(M_PI * x_Value) * std::sin(M_PI * y_Value) * std::sin(M_PI * y_Value) * std::cos(M_PI * gamma * t_value) * std::cos(M_PI * gamma * t_value)) ;
}

double initial_cond(double x_Value, double y_Value) {
    return pow((std::sin(M_PI * x_Value)), 2)*pow((std::sin(M_PI * y_Value)), 2);
}

double source_func(double x_value, double y_value, double t_value, double prev_value, double beta, double gamma, double alpha) {
    double temp_value = (
            -2 * M_PI * std::sin(M_PI * x_value) * std::sin(M_PI * x_value) * std::cos(M_PI * y_value)* std::cos(M_PI * y_value) * std::sin(M_PI * gamma * t_value) * std::cos(M_PI * gamma * t_value)
            -2 * alpha * M_PI * M_PI * (std::cos(M_PI * x_value) * std::cos(M_PI * x_value) - std::sin(M_PI * x_value) * std::sin(M_PI * x_value)) * std::sin(M_PI * y_value) * std::sin(M_PI * y_value) * std::cos(M_PI * t_value * gamma) * std::cos(M_PI * t_value * gamma)
            -2 * alpha * M_PI * M_PI * (std::cos(M_PI * y_value) * std::cos(M_PI * y_value) - std::sin(M_PI * y_value) * std::sin(M_PI * y_value)) * std::sin(M_PI * x_value) * std::sin(M_PI * x_value) * std::cos(M_PI * t_value * gamma) * std::cos(M_PI * t_value * gamma)
            - beta * prev_value * prev_value * prev_value * prev_value
        );
    return temp_value;
}

double source_func_linear(double x_value, double y_value, double t_value, double beta, double gamma, double alpha) {
    double temp_value = (
            -2 * M_PI * std::sin(M_PI * x_value) * std::sin(M_PI * x_value) * std::cos(M_PI * y_value)* std::cos(M_PI * y_value) * std::sin(M_PI * gamma * t_value) * std::cos(M_PI * gamma * t_value)
            -2 * alpha * M_PI * M_PI * (std::cos(M_PI * x_value) * std::cos(M_PI * x_value) - std::sin(M_PI * x_value) * std::sin(M_PI * x_value)) * std::sin(M_PI * y_value) * std::sin(M_PI * y_value) * std::cos(M_PI * t_value * gamma) * std::cos(M_PI * t_value * gamma)
            -2 * alpha * M_PI * M_PI * (std::cos(M_PI * y_value) * std::cos(M_PI * y_value) - std::sin(M_PI * y_value) * std::sin(M_PI * y_value)) * std::sin(M_PI * x_value) * std::sin(M_PI * x_value) * std::cos(M_PI * t_value * gamma) * std::cos(M_PI * t_value * gamma)
        );  
    return temp_value;      
}

void write_to_file(const Eigen::MatrixXd& matrix, int timestep, int write_interval) {
    if (timestep % write_interval != 0) {
        return;
    }

    // Write to VTK file
    {
        // Create a VTK structured points object
        vtkSmartPointer<vtkStructuredPoints> structuredPoints = vtkSmartPointer<vtkStructuredPoints>::New();
        structuredPoints->SetDimensions(matrix.rows(), matrix.cols(), 1);
        structuredPoints->SetOrigin(0.0, 0.0, 0.0);
        structuredPoints->SetSpacing(1.0, 1.0, 1.0);

        // Create a VTK array to store the data
        vtkSmartPointer<vtkDoubleArray> dataArray = vtkSmartPointer<vtkDoubleArray>::New();
        dataArray->SetNumberOfComponents(1);
        dataArray->SetNumberOfTuples(matrix.rows() * matrix.cols());

        // Fill the VTK array with data from the matrix
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                double value = matrix(i, j);
                dataArray->SetValue(i * matrix.cols() + j, value);
            }
        }

        // Assign the VTK array to the structured points object
        structuredPoints->GetPointData()->SetScalars(dataArray);

        // Write the structured points to a VTK file
        vtkSmartPointer<vtkStructuredPointsWriter> writer = vtkSmartPointer<vtkStructuredPointsWriter>::New();
        std::string vtk_filename = "results/results_timestep_" + std::to_string(timestep) + ".vtk";
        writer->SetFileName(vtk_filename.c_str());
        writer->SetInputData(structuredPoints);
        writer->Write();
    }

    // Write to TXT file
    {
        std::string txt_filename = "results/results_timestep_" + std::to_string(timestep) + ".txt";
        std::ofstream txt_file(txt_filename);

        if (!txt_file.is_open()) {
            std::cerr << "Unable to open TXT file for writing." << std::endl;
            return;
        }

        // Write matrix data to the TXT file
        
        txt_file << matrix << std::endl;

        txt_file.close();
    }
}

void compute_L2_norm(const Eigen::MatrixXd& numeric, const Eigen::MatrixXd& analytic, int timestep) {
    double L2_norm = (numeric - analytic).norm() / std::sqrt(numeric.size());
    
    // Write L2 norm to TXT file
    std::string l2_filename = "results/L2_norm_timestep_" + std::to_string(timestep) + ".txt";
    std::ofstream l2_file(l2_filename);
    
    if (!l2_file.is_open()) {
        std::cerr << "Unable to open L2 norm file for writing." << std::endl;
        return;
    }

    l2_file << "Timestep: " << timestep << ", L2 Norm: " << L2_norm << std::endl;
    l2_file.close();
}