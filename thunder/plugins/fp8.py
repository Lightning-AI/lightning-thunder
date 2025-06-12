from thunder import Plugin


class FP8(Plugin):
    """
    Plugin for enabling FP8 precision via NVIDIA Transformer Engine, enabling higher throughput of matrix operations in FP8.

    See `lightning-thunder/thunder/executors/transformer_engineex.py` for implementation details.
    """

    def setup_executors(self):
        """
        Imports the Transformer Engine executor.

        Returns:
            list[Executor]: A list containing the Transformer Engine executor.

        """
        from thunder.executors.transformer_engineex import transformer_engine_ex

        return [transformer_engine_ex]
