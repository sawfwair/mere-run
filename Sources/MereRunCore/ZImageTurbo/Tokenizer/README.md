# Qwen tokenizer

`QwenTokenizer` loads the checkpoint tokenizer and chat template for Qwen text
and image requests. The runtime uses it to render messages and encode special
tokens such as the end of a thinking block.

Token-by-token text delivery uses the incremental byte decoder when the
checkpoint declares a ByteLevel decoder and disables tokenization-space cleanup.
Other tokenizers decode the generated prefix for streaming progress.
