# MereRunEnvironment

`MereRunEnvironmentSwitch` is the one parser for `MERERUN_*` on/off switches.
`1`, `true`, `yes`, and `on` enable a switch; `0`, `false`, `no`, and `off`
disable it, ignoring case and surrounding whitespace. An unset, empty, or
unrecognized value keeps the caller's default.

The target has no dependencies so every runtime and model target can read
switches the same way. Read a switch with `MereRunEnvironmentSwitch.isEnabled`
or, when unset must stay distinct from off, `MereRunEnvironmentSwitch.value`.
Don't compare a switch's raw value by hand: `MereRunEnvironmentRegistryTests`
rejects subscripting a registered switch in production code.

Variables that select a mode or a number, such as `MERERUN_Q35_PRISM_FUSION`
or `MERERUN_Q38_DECODE_ASYNC_BLOCKS`, are not switches and keep their own
parsing.
