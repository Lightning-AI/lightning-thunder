from thunder import Plugin


class FP8(Plugin):
    def setup_executors(self):
        from thunder.executors.transformer_engineex import transformer_engine_ex

        return [transformer_engine_ex]
