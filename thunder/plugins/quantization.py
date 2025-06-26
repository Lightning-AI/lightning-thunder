from thunder import Plugin


class QuantizeInt4(Plugin):
    """
    Plugin for 4-bit integer quantization using BitsAndBytes.

    This plugin applies a 4-bit linear quantization transform to
    model weights, reducing memory footprint and improving
    throughput for both training and inference.

    See https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py#L889 for more details.
    """

    def setup_transforms(self):
        """
        Fetches the BitsAndBytes quantization transform.

        Returns:
            list[Transform]: A list containing the Transformer Engine executor.
        """

        from thunder.transforms.quantization import BitsAndBytesLinearQuant4bit

        return [BitsAndBytesLinearQuant4bit()]

    def setup_executors(self):
        """
        Fetches the BitsAndBytes quantization executor.

        Returns:
            list[Executor]: A list containing the Transformer Engine executor.

        """
        from thunder.transforms.quantization import get_bitsandbytes_executor

        return [get_bitsandbytes_executor()]
