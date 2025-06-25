Logging in Thunder
##################

.. note::
   Thunder logging is in its early stage of development. Currently, logging is only available around executors. We plan to extend logging support to other components like ThunderFX in future releases.

Thunder provides configurable logging through the ``THUNDER_LOGS`` environment variable.

Configuring Logging
===================

You can control Thunder's logging behavior by setting the ``THUNDER_LOGS`` environment variable. This is particularly useful for debugging and understanding what's happening inside Thunder's executors.

Basic Usage
-----------

To enable logging, set the ``THUNDER_LOGS`` environment variable before running your Python script::

  THUNDER_LOGS="info" python my_script.py

Supported Values
----------------

The ``THUNDER_LOGS`` environment variable supports the following values:

* Standard logging levels:
  * ``debug``: Set logging level to DEBUG
  * ``info``: Set logging level to INFO
  * ``warning``: Set logging level to WARNING
  * ``error``: Set logging level to ERROR
  * ``critical``: Set logging level to CRITICAL

* Executor-specific logging:
  * ``executors``: Allow all executors to log
  * ``<executor name>``: Allow a specific executor to log (e.g., ``nvfuser``, ``sdpa``)

* Prefix with ``+`` to set DEBUG level for an executor:
  * ``+<executor name>``: Set logging level to DEBUG for the specified executor
  * Without ``+``, the logging level defaults to WARNING

Multiple values can be specified by separating them with commas.

Examples
--------

Enable DEBUG level logging for the nvFuser executor and WARNING level for the SDPA executor::

  THUNDER_LOGS="+nvfuser,sdpa" python my_script.py

Set the global logging level to INFO::

  THUNDER_LOGS="info" python my_script.py

Get help about available logging options::

  THUNDER_LOGS="help" python my_script.py

Internal Implementation
=======================

Thunder uses Python's standard logging module internally. When you set the ``THUNDER_LOGS`` environment variable, Thunder configures the appropriate logging levels and filters based on your settings.

Each executor in Thunder has its own logger, and the logging system filters messages based on the executor names you specify in the ``THUNDER_LOGS`` environment variable.
