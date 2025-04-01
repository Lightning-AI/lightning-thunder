from thunder import Plugin


class QuantizeInt4(Plugin):
    def setup_transforms(self):
        from thunder.transforms.quantization import BitsAndBytesLinearQuant4bit

        return [BitsAndBytesLinearQuant4bit()]

    def setup_executors(self):
        from thunder.transforms.quantization import get_bitsandbytes_executor

        return [get_bitsandbytes_executor()]
