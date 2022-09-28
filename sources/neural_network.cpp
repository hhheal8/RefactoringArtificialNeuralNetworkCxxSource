#include <algorithm>
#include <iostream>
#include <math.h>
#include <random>
// #include <string>
#include <vector>

#include "neural_network.hpp"
#include "dealloc_all.hpp"
#include "layer.hpp"
#include "matrix.hpp"
#include "var_type_alias.hpp"

neural_network::neural_network(vec1d_szt topology, double bias, double learning_rate, double momentum):
m_topology(topology), m_bias(bias), m_learning_rate(learning_rate), m_momentum(momentum) {

  m_topology_size = m_topology.size();
  
  for(size_t i{}; i < m_topology_size; ++i) {
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
      m_layers.emplace_back(i);
      dealloc_all(l);
    }
  }

  for(size_t i{}; i < (m_topology_size - 1); ++i) {
    matrix *weight_matrix = new matrix(m_topology.at(i), m_topology.at(i + 1), true);
    m_weight_matrices.emplace_back(weight_matrix);
    dealloc_all(weight_matrix);
  }

  for(size_t i{}; i < m_topology.at((m_topology_size - 1)); ++i) {
    m_errors.emplace_back(0.00);
  }

  m_error = 0.00;

}

neural_network::neural_network(
  vec1d_szt topology, 
  size_t hidden_activation_type, 
  size_t output_activation_type, 
  size_t cost_fun_type, 
  double bias, 
  double learning_rate, 
  double momentum
): m_topology(topology), 
  m_hidden_activation_type(hidden_activation_type), m_output_activation_type(output_activation_type),
  m_cost_fun_type(cost_fun_type), m_bias(bias), 
  m_learning_rate(learning_rate), m_momentum(momentum) {
  
  for(size_t i{}; i < m_topology_size; ++i) {
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
      m_layers.emplace_back(i);
      dealloc_all(l);
    }
  }

  for(size_t i{}; i < (m_topology_size - 1); ++i) {
    matrix *weight_matrix = new matrix(m_topology.at(i), m_topology.at(i + 1), true);
    m_weight_matrices.emplace_back(weight_matrix);
    dealloc_all(weight_matrix);
  }

  for(size_t i{}; i < m_topology.at((m_topology_size - 1)); ++i) {
    m_errors.emplace_back(0.00);
  }

  m_error = 0.00;

}

auto neural_network::set_current_input(vec1d_dbl input) -> void {
  m_input = input;
  for(size_t i{}; i < m_input.size(); ++i) {
    m_layers.at(0)->set_val(i, m_input.at(i));
  }
}