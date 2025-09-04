Logging in Thunder
##################

.. note::
   Thunder logging is in its early stage of development. Currently, logging is only available around executors. We plan to extend logging support to other components like ThunderFX in future releases.

Thunder provides configurable logging through the ``THUNDER_LOGS`` environment variable.

Configuring Logging
===================

You can control Thunder's logging behavior by setting the ``THUNDER_LOGS`` environment variable.

Basic Usage
-----------

To enable logging, set the ``THUNDER_LOGS`` environment variable before running your Python script::

  THUNDER_LOGS="+nvfuser" python my_script.py

Supported Values
----------------

The ``THUNDER_LOGS`` environment variable supports the following values:

* Special options:
  * ``traces``: Enable logging of Thunder traces from :func:`thunder.jit`

* Executor-specific logging:
  * ``executors``: Allow all executors to log
  * ``<executor name>``: Allow a specific executor to log (e.g., ``nvfuser``, ``sdpa``)

* Executor logging levels can be controlled with prefixes:
  * ``<executor name>`` (no prefix): Set logging level to INFO for the specified executor
  * ``+<executor name>``: Set logging level to DEBUG for the specified executor
  * ``-<executor name>``: Set logging level to WARNING for the specified executor

Multiple values can be specified by separating them with commas.

Examples
--------

**Component-specific logging:**

Enable different logging levels for various executors::

  THUNDER_LOGS="+nvfuser,-torch_compile,sdpa" python my_script.py

This will set DEBUG level for nvFuser, WARNING level for torch_compile, and INFO level for SDPA.

**Help:**

Get help about available logging options::

  THUNDER_LOGS="help" python my_script.py

**Show traces:**

Enable logging of Thunder traces::

  THUNDER_LOGS="traces" python my_script.py

Internal Implementation
=======================

Thunder uses Python's standard logging module internally. When you set the ``THUNDER_LOGS`` environment variable, Thunder configures the appropriate logging levels and filters based on your settings.

Each executor in Thunder has its own logger, and the logging system filters messages based on the executor names you specify in the ``THUNDER_LOGS`` environment variable.
