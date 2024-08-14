.. module:: thunder

thunder
=======


Compiling functions and modules
-------------------------------


.. autosummary::
    :toctree: generated/

    jit
    functional.jit


Querying information on compiled functions and modules
------------------------------------------------------


.. autosummary::
    :toctree: generated/

    compile_data
    compile_stats
    last_traces
    last_backward_traces
    last_prologue_traces
    cache_option
    cache_hits
    cache_misses
    list_transforms
    last_interpreted_instructions
    last_interpreter_log
    last_compile_options
..
    compile
    grad

JITed Model wrapper
-------------------

.. autoclass:: ThunderModule
    :members: no_sync
    :exclude-members: forward
