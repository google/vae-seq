import sonnet as snt

class FeedbackRNNCore(snt.RNNCore):
    """Wraps an RNN Core to make the input depend on the previous output."""

    def __init__(self, core, encode, decode, name=None):
        """
        Args:
          core: The inner RNNCore
          encode: inputs, feedback_state -> inner_input, decode_state
          decode: inner_output, decode_state -> output, feedback_state
        """
        super(FeedbackRNNCore, self).__init__(name or self.__class__.__name__)
        self._core = core
        self._encode = encode
        self._decode = decode

    @property
    def state_size(self):
        return (self._core.state_size, self._decode.output_size[1])

    @property
    def output_size(self):
        return self._decode.output_size[0]

    def _build(self, inputs, (inner_state, feedback_state)):
        inner_input, decode_state = self._encode(inputs, feedback_state)
        inner_output, inner_state = self._core(inner_input, inner_state)
        output, feedback_state = self._decode(inner_output, decode_state)
        return output, (inner_state, feedback_state)
