#include <iostream>
#include <math.h>
#include <random>
#include <vector>

#include "neuron.hpp"

neuron::neuron(double val) {
  set_val(val);
}

neuron::neuron(double val, int activation_type): m_activation_type(activation_type) {
  set_val(val);
}

auto neuron::set_val(double val) -> void {
  m_val = val;
  activate();
  derive();
}

auto neuron::activate() -> void {

  if(m_activation_type == TANH) {
    m_activated_val = tanh(m_val);
  }
  else if(m_activated_val == RELU) {
    if(m_val > 0) {
      m_activated_val = m_val;
    }
    else {
      m_activated_val = 0;
    }
  }
  else if(m_activated_val == SIGM) {
    m_activated_val = (1 / (1 + exp(- m_val)));
  }
  else {
    m_activated_val = (1 / (1 + exp(- m_val)));
  }

}

auto neuron::derive() -> void {

  if(m_activation_type == TANH) {
    m_derived_val = (1.0 - (m_activated_val * m_activated_val));
  }
  else if(m_activated_val == RELU) {
    if(m_val > 0) {
      m_derived_val = 1;
    }
    else {
      m_derived_val = 0;
    }
  }
  else if(m_activated_val == SIGM) {
    m_derived_val = (m_activated_val * (1 - m_activated_val));
  }
  else {
    m_derived_val = (m_activated_val * (1 - m_activated_val));
  }

}
