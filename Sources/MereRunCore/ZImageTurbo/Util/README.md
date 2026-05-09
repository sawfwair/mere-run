# ZImageTurbo Util

Small utility code used by ZImage Turbo.

- `ImageIO.swift`: image loading and writing helpers for generation paths.

Keep utility code narrow. If a helper becomes model- or CLI-specific, move it
closer to that caller.
