#!/usr/bin/env scons

"""
The master SConstruct file; particular analyses run via SConscript files in subdirectories.
"""

import os
from os.path import join
import SCons.Script as sc


# Set up environment and run  \m/ ( -_- ) \m/
sc.AddOption(
    '--test',
    type='string',
    help="Which test to run.",
    default='simulation')

env = sc.Environment(ENV=os.environ)
sc.Export('env')
sc.SConscript(join(sc.GetOption('test'), 'SConscript'))
