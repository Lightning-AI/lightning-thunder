from thunder import Plugin


class FP8(Plugin):
    """
    Plugin for enabling FP8 precision via NVIDIA Transformer Engine, enabling higher throughput of matrix operations in FP8.

    See `lightning-thunder/thunder/executors/transformer_engineex.py` for implementation details.
    """

    def setup_transforms(self):
        """
        Fetches the TransformerEngine transform.

        Returns:
            list[Transform]: A list containing the TransformerEngine transforms.
        """

        from thunder.executors.transformer_engineex import TransformerEngineTransform

        # When TE executor is not available, both the transform and the executor will be None.
        if TransformerEngineTransform is None:
            return []

        return [TransformerEngineTransform()]

    def setup_executors(self):
        """
        Imports the TransformerEngine executor.

        Returns:
            list[Executor]: A list containing the Transformer Engine executor.

        """
        from thunder.executors.transformer_engineex import transformer_engine_ex

        # When TE executor is not available, both the transform and the executor will be None.
        if transformer_engine_ex is None:
            return []

        return [transformer_engine_ex]
