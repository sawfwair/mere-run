# MuScriptor

Native MLX inference for the open MuScriptor audio-to-MIDI checkpoints from
Kyutai and Mirelo.

The runtime consumes mono 16 kHz audio in five-second chunks, reproduces the
published 512-bin HTK mel frontend, prepends the learned dataset/instrument
conditioning tokens, and autoregressively decodes the MT3 event vocabulary.
The decoder carries tied notes across chunk boundaries and emits either note
events or a Standard MIDI File with one track per detected instrument.

Published weights are gated on Hugging Face and licensed CC BY-NC 4.0. Users
must accept the model terms and configure a Hugging Face token before pulling.
