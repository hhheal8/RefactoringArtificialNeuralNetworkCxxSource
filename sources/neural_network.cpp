#include <assert.h>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <random>
// #include <string>
#include <vector>

#include "neural_network.hpp"
#include "ANN_math.hpp"
#include "dealloc_all.hpp"
#include "layer.hpp"
#include "matrix.hpp"
#include "neuron.hpp"
#include "var_type_alias.hpp"

neural_network::neural_network(vec1d_int topology, double bias, double learning_rate, double momentum):
m_topology(topology), m_bias(bias), m_learning_rate(learning_rate), m_momentum(momentum) {

  m_topology_size = m_topology.size();
  
  for(int i{}; i < m_topology_size; ++i) {
    if(i > 0 && i < (m_topology_size - 1)) {
      layer *l = new layer(m_topology.at(i), m_hidden_activation_type);
      m_layers.emplace_back(l);
      dealloc_all(l);
    }
    else if(i == (m_topology_size - 1)) {
      layer *l = new layer(m_topology.at(i), m_output_activation_type);
      dealloc_all(l);
    }
    else {
      layer *l = new layer(m_topology.at(i));
      m_layers.emplace_back(l);
      dealloc_all(l);
    }
  }

  for(int i{}; i < (m_topology_size - 1); ++i) {
    matrix *weight_matrix = new matrix(m_topology.at(i), m_topology.at(i + 1), true);
    m_weight_matrices.emplace_back(weight_matrix);
    dealloc_all(weight_matrix);
  }

  for(int i{}; i < m_topology.at((m_topology_size - 1)); ++i) {
    m_errors.emplace_back(0.00);
  }

  m_error = 0.00;

}

neural_network::neural_network(
  vec1d_int topology, 
  int hidden_activation_type, 
  int output_activation_type, 
  int cost_fun_type, 
  double bias, 
  double learning_rate, 
  double momentum
): m_topology(topology), 
  m_hidden_activation_type(hidden_activation_type), m_output_activation_type(output_activation_type),
  m_cost_fun_type(cost_fun_type), m_bias(bias), 
  m_learning_rate(learning_rate), m_momentum(momentum) {
  
  for(int i{}; i < m_topology_size; ++i) {
    if(i > 0 && i < (m_topology_size - 1)) {
      layer *l = new layer(m_topology.at(i), m_hidden_activation_type);
      m_layers.emplace_back(l);
      dealloc_all(l);
    }
    else if(i == (m_topology_size - 1)) {
      layer *l = new layer(m_topology.at(i), m_output_activation_type);
      dealloc_all(l);
    }
    else {
      layer *l = new layer(m_topology.at(i));
      m_layers.emplace_back(l);
      dealloc_all(l);
    }
  }

  for(int i{}; i < (m_topology_size - 1); ++i) {
    matrix *weight_matrix = new matrix(m_topology.at(i), m_topology.at(i + 1), true);
    m_weight_matrices.emplace_back(weight_matrix);
    dealloc_all(weight_matrix);
  }

  for(int i{}; i < m_topology.at((m_topology_size - 1)); ++i) {
    m_errors.emplace_back(0.00);
  }

  m_error = 0.00;

}

auto neural_network::set_current_input(vec1d_dbl input) -> void {
  m_input = input;
  for(int i{}; i < m_input.size(); ++i) {
    m_layers.at(0)->set_val(i, m_input.at(i));
  }
}

auto neural_network::feed_forward() -> void {

  matrix *a; //Matrix of neurons to the left
  matrix *b; //Matrix of weights to the left of layer
  matrix *c; //Matrix of neurons to the next layer

  for(int i{}; i < (m_topology_size - 1); ++i) {
    a = get_neuron_matrix(i);
    b = get_weight_matrix(i);
    c = new matrix(a->get_row(), b->get_col(), false); 

    if(i != 0) {
      a = get_activated_neuron_matrix(i);
    }

    utils::ANN_math::multiply_matrix(a, b, c);

    for(int c_i{}; c_i < c->get_row(); ++c_i) {
      set_neuron_val(i + 1, c_i, c->get_val(0, c_i) + m_bias);
    }

    dealloc_all(a, b, c);
  }

}

auto neural_network::back_propagation() -> void {

  vec1d_matrix_p new_weights;

  matrix *delta_weights;
  matrix *gradients;
  matrix *derived_values;
  matrix *gradients_transpose;
  matrix *z_activated_val;
  matrix *temp_new_weights;
  matrix *p_gradients;
  matrix *transpose_p_weights;
  matrix *hidden_derived;
  matrix *transposed_hidden;
  
  //PART 1: Output to last hidden layer
  int index_output_layer = m_topology.size() - 1;

  gradients = new matrix(1, m_topology.at(index_output_layer), false);

  derived_values = m_layers.at(index_output_layer)->matrixify_derived_vals();

  for(int i{}; i < m_topology.at(index_output_layer); ++i) {
    double e = m_derived_errors.at(i);
    double y = derived_values->get_val(0, i);
    double g = e * y;
    gradients->set_val(0, i, g);
  }
  
  //Gt * Z
  gradients_transpose = gradients->transpose();

  z_activated_val = m_layers.at(index_output_layer - 1)->matrixify_activated_vals();
  
  delta_weights = new matrix(gradients_transpose->get_row(), z_activated_val->get_col(), false);

  utils::ANN_math::multiply_matrix(gradients_transpose, z_activated_val, delta_weights);

  //Compute for new weights (last hidden <-> output)
  temp_new_weights = new matrix(m_topology.at(index_output_layer - 1), m_topology.at(index_output_layer), false);

  for(int r{}; m_topology.at(index_output_layer - 1); ++r) {
    for(int c{}; m_topology.at(index_output_layer); ++c) {
      double original_weight_val = m_weight_matrices.at(index_output_layer - 1)->get_val(r, c);
      double delta_val = delta_weights->get_val(r, c);
      
      original_weight_val = m_momentum * original_weight_val;
      delta_val = m_learning_rate * delta_val;

      temp_new_weights->set_val(r, c, (original_weight_val - delta_val));
    }
  }

  new_weights.emplace_back(new matrix(*temp_new_weights));

  //FIXME:
  dealloc_all(gradients_transpose, z_activated_val, temp_new_weights, delta_weights, derived_values);

  //Hidden output layer
  for(int i{(index_output_layer - 1)}; i > 0; --i) {

    p_gradients = new matrix(*gradients);
    dealloc_all(p_gradients);
    
    transpose_p_weights = m_weight_matrices.at(i)->transpose();

    gradients = new matrix(p_gradients->get_row(), transpose_p_weights->get_col(), false);

    utils::ANN_math::multiply_matrix(p_gradients, transpose_p_weights, gradients);

    hidden_derived = m_layers.at(i)->matrixify_derived_vals();

    for(int c_counter{}; c_counter < hidden_derived->get_row(); ++c_counter) {
      double g = gradients->get_val(0, c_counter) * hidden_derived->get_val(0, c_counter);
      gradients->set_val(0, c_counter, g);
    }

    if(i == 1) {
      z_activated_val = m_layers.at(0)->matrixify_vals();
    }
    else {
      z_activated_val = m_layers.at(i - 1)->matrixify_activated_vals();
    }

    transposed_hidden = z_activated_val->transpose();

    delta_weights = new matrix(transposed_hidden->get_row(), gradients->get_col(), false);

    utils::ANN_math::multiply_matrix(transposed_hidden, gradients, delta_weights);

    //Update weights
    temp_new_weights = new matrix(m_weight_matrices.at(i - 1)->get_row(), m_weight_matrices.at(i - 1)->get_col(), false);

    for(int r{}; temp_new_weights->get_row(); ++r) {
      for(int c{}; temp_new_weights->get_col(); ++c) {
        double original_weight_val = m_weight_matrices.at(i - 1)->get_val(r, c);
        double delta_val = delta_weights->get_val(r, c);
        
        original_weight_val = m_momentum * original_weight_val;
        delta_val = m_learning_rate * delta_val;

        temp_new_weights->set_val(r, c, (original_weight_val - delta_val));
      }
    }

    new_weights.emplace_back(new matrix(*temp_new_weights));

    dealloc_all(p_gradients, transpose_p_weights, hidden_derived, z_activated_val, transposed_hidden, temp_new_weights, delta_weights);

  }
  dealloc_all(gradients);

  for(int i{}; i < m_weight_matrices.size(); ++i) {
    delete m_weight_matrices[i];
  }
  m_weight_matrices.clear();

  std::reverse(new_weights.begin(), new_weights.end());

  for(int i{}; i < new_weights.size(); ++i) {
    m_weight_matrices.emplace_back(new matrix(*new_weights[i]));
    delete new_weights[i];
  }

}

auto neural_network::set_error_MSE() -> void {

  int output_layer_index = m_layers.size() - 1;
  std::vector<neuron*> output_neurons = m_layers.at(output_layer_index)->get_neuron();

  m_error = 0.00;

  for(int i{}; i < m_target.size(); ++i) {
    double t = m_target.at(i);
    double y = output_neurons.at(i)->get_derived_val();

    m_errors.at(i) = 0.5 * pow(abs((t - y)), 2);
    m_derived_errors.at(i) = (y - t);

    m_error += m_errors.at(i);
  }

}

auto neural_network::set_errors() -> void {

  // if(m_target.size() == 0) {
  //   std::cerr << "\nNo target for this Neural Network\n";
  //   assert(false);
  // }
  // if(m_target.size() != m_layers.at(m_layers.size() - 1)->get_neuron().size()) {
  //   std::cerr << "\nTarget size (" << m_target.size() << ") is not the same as output layer size: " 
  //             << m_layers.at(m_layers.size() - 1)->get_neuron().size() << "\n";

  //   for(int i{}; i < m_target.size(); ++i) {
  //     std::cout << m_target.at(i) << "\n";
  //   }
  //   assert(false);
  // }
  switch(m_cost_fun_type) {
    case(COST_MSE): 
      set_error_MSE();
    break;

    default:
      set_error_MSE();
    break;
  }

}