# Scenario file format

A scenario file is JSON with a receiver and a list of emitters. It can be used anywhere
a built-in scenario is accepted and is copied into exported experiment reports.

## Root fields

- `name`: optional non-empty report label;
- `num_bands`: number of coarse receiver bands;
- `duration`: number of simulation steps;
- `receiver`: optional receiver settings;
- `emitters`: array of emitter definitions.

Unknown root or constructor fields are rejected so spelling mistakes do not silently
change an experiment.

## Receiver

Receiver fields match the receiver model: detection and false-alarm probabilities,
sensitivity, power and pulse-width noise, retuning delay, and tuning speed.
`seed_offset` is added to the experiment seed. This gives each repeated run new but
reproducible receiver errors.

## Emitters

Supported `type` values are:

- `periodic`;
- `frequency-hopping`;
- `scanning`;
- `burst`;
- `jittered-periodic`;
- `windowed`, containing one nested `emitter`;
- `mode-switching`, containing `first_mode` and `second_mode`.

The remaining fields match the corresponding emitter model. A phase may be an integer
or the string `"random"`. Random phases use the experiment seed. Jittered emitters may
specify `seed_offset` for an independent deterministic sequence.

Mode definitions must use the same `emitter_id` on both sides of the switch. Band and
range validation is performed by the simulation before an experiment begins.

See [examples/custom-scenario.json](../examples/custom-scenario.json) for a runnable
definition using fixed, hopping, and mode-switching emitters.
