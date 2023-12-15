# -*- coding: utf-8 -*-
"""
Global attributes for bending modules.

@author: MarcGebhardt
"""

# significant digits preset (float precision handling)
# (use with exmecheva.common.helper.round_to_sigdig(x, _significant_digits_fpd))
_significant_digits_fpd=12

# set Parameter types, which are not marked as free for fit:
_param_types_not_free   = ['independent','expr','fixed','post']
_param_types_fit_or_set = ['free','post']