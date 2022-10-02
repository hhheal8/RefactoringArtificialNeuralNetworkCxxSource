#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#define COST_MSE 1

#include <assert.h>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <random>
// #include <string>
#include <vector>

#include "ANN_math.hpp"
#include "dealloc_all.hpp"
#include "layer.hpp"
#include "matrix.hpp"
#include "neuron.hpp"
#include "var_type_alias.hpp"

class ANN_math;
class layer;
class matrix;
class neuron;

class neural_network {

  public:

    int m_topology_size{};
    int m_hidden_activation_type{RELU};
    int m_output_activation_type{SIGM};
    int m_cost_fun_type{COST_MSE};

    vec1d_int m_topology;

    using vec1d_layer_p = std::vector<layer*>; //REVIEW: layer_p - layer*
    using vec1d_matrix_p = std::vector<matrix*>; //REVIEW: matrix_p - matrix*

    vec1d_layer_p m_layers;
    vec1d_matrix_p m_weight_matrices;
    vec1d_matrix_p m_gradient_matrices; 

    vec1d_dbl m_input;
    vec1d_dbl m_target;
    vec1d_dbl m_errors;
    vec1d_dbl m_derived_errors;

    double m_error{0};
    double m_bias{1};
    double m_momentum{};
    double m_learning_rate{};

  public:
    
    neural_network(
      vec1d_int topology, 
      double bias,
      double learning_rate,
      double momentum
    );

    neural_network(
      vec1d_int topology, 
      int hidden_activation_type,
      int output_activation_type,
      int cost_fun_type,
      double bias,
      double learning_rate,
      double momentum
    );

    auto set_current_input(vec1d_dbl input) -> void;
    auto set_current_target(vec1d_dbl target) -> void { m_target = target; }

    auto feed_forward() -> void;
    auto back_propagation() -> void;
    auto set_errors() -> void;

    auto get_activated_vals(int index) const -> vec1d_dbl { return m_layers.at(index)->get_activated_vals(); }

    matrix *get_neuron_matrix(int index) { return m_layers.at(index)->matrixify_vals(); }

    matrix *get_activated_neuron_matrix(int index) { return m_layers.at(index)->matrixify_activated_vals(); }
    matrix *get_derived_neuron_matrix(int index) { return m_layers.at(index)->matrixify_derived_vals(); }
    matrix *get_weight_matrix(int index) { return new matrix(*this->m_weight_matrices.at(index)); }

    auto set_neuron_val(int i_layer, int i_neuron, double val) -> void { m_layers.at(i_layer)->set_val(i_neuron, val); };

  private:

    auto set_error_MSE() -> void;

};

#endif // NEURAL_NETWORK_H