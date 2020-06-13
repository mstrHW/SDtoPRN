"""
Python model "pp_model.py"
Translated using PySD version 0.8.3
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

__pysd_version__ = "0.8.3"


@cache('step')
def predator_births():
    """
    predator births



    component


    """
    return np.exp(
        functions.log(predator_population(), np.exp(1)) +
        functions.log(predator_birth_fraction(), np.exp(1)) +
        functions.log(prey_population(), np.exp(1)))


@cache('run')
def predator_birth_fraction():
    """
    predator birth fraction

    [0,0.05,0.001]

    constant


    """
    return 0.01


@cache('run')
def predator_death_proportionality_constant():
    """
    predator death proportionality constant

    [0,2,0.05]

    constant


    """
    return 1.05


@cache('step')
def predator_deaths():
    """
    predator deaths



    component


    """
    return np.exp(
        functions.log(predator_death_proportionality_constant(), np.exp(1)) +
        functions.log(predator_population(), np.exp(1)))


@cache('step')
def predator_population():
    """
    Predator Population



    component


    """
    return integ_predator_population()


@cache('run')
def prey_birth_fraction():
    """
    prey birth fraction

    [0,5,0.1]

    constant


    """
    return 2


@cache('step')
def prey_births():
    """
    prey births



    component


    """
    return np.exp(
        functions.log(prey_birth_fraction(), np.exp(1)) +
        functions.log(prey_population(), np.exp(1)))


@cache('run')
def prey_death_proportionality_constant():
    """
    prey death proportionality constant

    [0,0.05,0.001]

    constant


    """
    return 0.02


@cache('step')
def prey_deaths():
    """
    prey deaths



    component


    """
    return np.exp(
        functions.log(predator_population(), np.exp(1)) +
        functions.log(prey_death_proportionality_constant(), np.exp(1)) +
        functions.log(prey_population(), np.exp(1)))


@cache('step')
def prey_population():
    """
    Prey Population



    component


    """
    return integ_prey_population()


@cache('run')
def final_time():
    """
    FINAL TIME

    seasons

    constant

    The final time for the simulation.
    """
    return 12


@cache('run')
def initial_time():
    """
    INITIAL TIME

    seasons

    constant

    The initial time for the simulation.
    """
    return 0


@cache('step')
def saveper():
    """
    SAVEPER

    seasons [0,?]

    component

    The frequency with which output is stored.
    """
    return time_step()


@cache('run')
def time_step():
    """
    TIME STEP

    seasons [0,?]

    constant

    The time step for the simulation.
    """
    return 0.03125


integ_predator_population = functions.Integ(lambda: predator_births() - predator_deaths(), lambda:
                                            15)

integ_prey_population = functions.Integ(lambda: prey_births() - prey_deaths(), lambda: 100)
