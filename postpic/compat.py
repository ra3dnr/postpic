#
# This file is part of postpic.
#
# postpic is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# postpic is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with postpic. If not, see <http://www.gnu.org/licenses/>.
#
# Stephan Kuschel 2017
'''
Compatibility module.
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


def _newfunction(x):
    return 2*x


np.uselessnewf = _newfunction


def mokeypatched_sin(*args, **kwargs):
    print(args)
    print(kwargs)
    return np_sin(*args, **kwargs)


np_sin = np.sin
np.sin = mokeypatched_sin
