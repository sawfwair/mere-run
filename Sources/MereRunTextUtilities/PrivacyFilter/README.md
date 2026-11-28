# PrivacyFilter

Local PII detection and anonymization support.

- `OpenAIPrivacyFilterConfig.swift`: typed model settings.
- `OpenAIPrivacyFilterTokenizer.swift`: tokenizer wrapper.
- `OpenAIPrivacyFilterDecoding.swift`: entity decoding.
- `OpenAIPrivacyFilterAnonymizer.swift`: replacement logic.

This is a security-sensitive path. Keep tests focused on entity spans,
replacement behavior, and compatibility with public CLI output.
