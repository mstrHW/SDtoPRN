"""
Python model "pp_model.py"
Translated using PySD version 0.10.0
"""
from __future__ import division
import numpy as np
from pysd import utils
import xarray as xr

from pysd.py_backend.functions import cache
from pysd.py_backend import functions

_subscript_dict = {}

_namespace = {
    'TIME': 'time',
    'Time': 'time',
    'predator births': 'predator_births',
    'predator birth fraction': 'predator_birth_fraction',
    'predator death proportionality constant': 'predator_death_proportionality_constant',
    'predator deaths': 'predator_deaths',
    'Predator Population': 'predator_population',
    'prey birth fraction': 'prey_birth_fraction',
    'prey births': 'prey_births',
    'prey death proportionality constant': 'prey_death_proportionality_constant',
    'prey deaths': 'prey_deaths',
    'Prey Population': 'prey_population',
    'FINAL TIME': 'final_time',
    'INITIAL TIME': 'initial_time',
    'SAVEPER': 'saveper',
    'TIME STEP': 'time_step'
}

__pysd_version__ = "0.10.0"

__data = {'scope': None, 'time': lambda: 0}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


def time():
    return __data['time']()


@cache('step')
def predator_births():
    """
    Real Name: b'predator births'
    Original Eqn: b'EXP(LOG(Predator Population, EXP(1)) + LOG(predator birth fraction, EXP(1)) + LOG(Prey Population\\\\ , EXP(1)))'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return np.exp(
        functions.log(predator_population(), np.exp(1)) +
        functions.log(predator_birth_fraction(), np.exp(1)) +
        functions.log(prey_population(), np.exp(1)))


@cache('run')
def predator_birth_fraction():
    """
    Real Name: b'predator birth fraction'
    Original Eqn: b'0.01'
    Units: b''
    Limits: (0.0, 0.05, 0.001)
    Type: constant

    b''
    """
    return 0.01


@cache('run')
def predator_death_proportionality_constant():
    """
    Real Name: b'predator death proportionality constant'
    Original Eqn: b'1.05'
    Units: b''
    Limits: (0.0, 2.0, 0.05)
    Type: constant

    b''
    """
    return 1.05


@cache('step')
def predator_deaths():
    """
    Real Name: b'predator deaths'
    Original Eqn: b'EXP(LOG(predator death proportionality constant, EXP(1)) + LOG(Predator Population, \\\\ EXP(1)))'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return np.exp(
        functions.log(predator_death_proportionality_constant(), np.exp(1)) +
        functions.log(predator_population(), np.exp(1)))


@cache('step')
def predator_population():
    """
    Real Name: b'Predator Population'
    Original Eqn: b'INTEG ( predator births-predator deaths, 15)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_predator_population()


@cache('run')
def prey_birth_fraction():
    """
    Real Name: b'prey birth fraction'
    Original Eqn: b'2'
    Units: b''
    Limits: (0.0, 5.0, 0.1)
    Type: constant

    b''
    """
    return 2


@cache('step')
def prey_births():
    """
    Real Name: b'prey births'
    Original Eqn: b'EXP(LOG(prey birth fraction, EXP(1)) + LOG(Prey Population, EXP(1)))'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return np.exp(
        functions.log(prey_birth_fraction(), np.exp(1)) +
        functions.log(prey_population(), np.exp(1)))


@cache('run')
def prey_death_proportionality_constant():
    """
    Real Name: b'prey death proportionality constant'
    Original Eqn: b'0.02'
    Units: b''
    Limits: (0.0, 0.05, 0.001)
    Type: constant

    b''
    """
    return 0.02


@cache('step')
def prey_deaths():
    """
    Real Name: b'prey deaths'
    Original Eqn: b'EXP(LOG(Predator Population, EXP(1)) + LOG(prey death proportionality constant, EXP(\\\\ 1)) + LOG(Prey Population, EXP(1)))'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return np.exp(
        functions.log(predator_population(), np.exp(1)) +
        functions.log(prey_death_proportionality_constant(), np.exp(1)) +
        functions.log(prey_population(), np.exp(1)))


@cache('step')
def prey_population():
    """
    Real Name: b'Prey Population'
    Original Eqn: b'INTEG ( prey births-prey deaths, 100)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_prey_population()


@cache('run')
def final_time():
    """
    Real Name: b'FINAL TIME'
    Original Eqn: b'12'
    Units: b'seasons'
    Limits: (None, None)
    Type: constant

    b'The final time for the simulation.'
    """
    return 12


@cache('run')
def initial_time():
    """
    Real Name: b'INITIAL TIME'
    Original Eqn: b'0'
    Units: b'seasons'
    Limits: (None, None)
    Type: constant

    b'The initial time for the simulation.'
    """
    return 0


@cache('step')
def saveper():
    """
    Real Name: b'SAVEPER'
    Original Eqn: b'TIME STEP'
    Units: b'seasons'
    Limits: (0.0, None)
    Type: component

    b'The frequency with which output is stored.'
    """
    return time_step()


@cache('run')
def time_step():
    """
    Real Name: b'TIME STEP'
    Original Eqn: b'0.03125'
    Units: b'seasons'
    Limits: (0.0, None)
    Type: constant

    b'The time step for the simulation.'
    """
    return 0.03125


_integ_predator_population = functions.Integ(lambda: predator_births() - predator_deaths(),
                                             lambda: 15)

_integ_prey_population = functions.Integ(lambda: prey_births() - prey_deaths(), lambda: 100)
