# mere.run for Pi

This Pi package registers a native `mere-run` provider and discovers compatible local models from the running mere.run API server. It reads the server's model task, tool-call support, reasoning levels, input modalities, context/output limits, and OpenAI compatibility flags instead of guessing from a model ID.

## Use it

Start mere.run first:

```bash
mere.run api serve --host 127.0.0.1 --port 8080
```

Install this package from a local checkout:

```bash
pi install /path/to/mere-run/integrations/pi
pi --provider mere-run
```

The package defaults to `http://127.0.0.1:8080/v1`. Override the endpoint or API key when needed:

```bash
MERERUN_BASE_URL=http://127.0.0.1:9090/v1 \
MERERUN_API_KEY=secret \
pi --provider mere-run
```

Only `chat.completions` models that report `tool_call: true` are exposed to Pi. Image, audio, embedding, geometry, and text-only chat models remain available through the mere.run API but are not presented as coding-agent models.

`mere.run agent start` remains the managed one-command path. It installs the current Pi release, starts the local server, writes an isolated provider extension with a selected-model fallback, and launches Pi.
