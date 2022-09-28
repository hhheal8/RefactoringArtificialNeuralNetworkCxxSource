#include <iostream>
#include <math.h>
#include <random>
#include <vector>

#include "layer.hpp"
#include "dealloc_all.hpp"
#include "neuron.hpp"
#include "matrix.hpp"
#include "var_type_alias.hpp"

layer::layer(size_t size): m_size(size) {
  
  for(size_t i{}; m_size; ++i) {
    
    neuron *n = new neuron(0.000000000);

    m_neurons.emplace_back(n);

    // if(i == m_neurons.size()) { //REVIEW:
      dealloc_all(n);
    // }

  }

}

layer::layer(size_t size, size_t activation_type): m_size(size) {
  
  for(size_t i{}; m_size; ++i) {
    
    neuron *n = new neuron(0.000000000, activation_type);

    m_neurons.emplace_back(n);

    // if(i == m_neurons.size()) { //REVIEW:
      dealloc_all(n);
    // }

  }

}

auto layer::set_val(size_t i_layer, double val) -> void {
  m_neurons.at(i_layer)->set_val(val);
}

auto layer::get_activated_vals() const -> vec1d_dbl {
  
  vec1d_dbl ret;

  for(size_t i{}; i < m_neurons.size(); ++i) {
    double val = m_neurons.at(i)->get_activated_val();
    ret.emplace_back(val);
  }

  return ret;

}

matrix *layer::matrixify_vals() {

  matrix *m = new matrix(1, m_neurons.size(), false);

  for(size_t i{}; i < m_neurons.size(); ++i) {
    m->set_val(0, 1, m_neurons.at(i)->get_val());
  }

  return m;

}

matrix *layer::matrixify_activated_vals() {

  matrix *m = new matrix(1, m_neurons.size(), false);

  for(size_t i{}; i < m_neurons.size(); ++i) {
    m->set_val(0, 1, m_neurons.at(i)->get_activated_val());
  }

  return m;

}

matrix *layer::matrixify_derived_vals() {

  matrix *m = new matrix(1, m_neurons.size(), false);

  for(size_t i{}; i < m_neurons.size(); ++i) {
    m->set_val(0, 1, m_neurons.at(i)->get_derived_val());
  }

  return m;

}