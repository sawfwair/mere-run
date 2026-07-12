# Native pose runtime

This module owns local body, hand, and face landmark extraction for VFX and
motion workflows. On Apple platforms it uses the system Vision framework and
returns typed, normalized landmark data. The CLI and companion plugins consume
these types; they do not own a separate pose inference runtime.
