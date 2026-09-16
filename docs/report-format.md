# Experiment report format

An experiment report preserves the inputs and aggregate results needed to repeat or
compare a simulation run. It deliberately excludes wall-clock timestamps and machine
details because they do not affect simulation output.

## Common metadata

Every report contains:

- `schema_version`: the report contract version;
- `scenario`: the selected built-in scenario;
- `scenario_definition`: the complete custom definition, or `null` for a built-in;
- `start_seed`: the first deterministic seed;
- `runs`: the number of consecutive seeds evaluated;
- `strategy_results`: one aggregate record per scheduler;
- `track_association`: aggregate association results for the track-aware scheduler.

The worker count is not stored because sequential and parallel execution are required
to produce identical results.

## JSON

JSON preserves the nested report structure. Strategy names map to their aggregate
metrics, while `track_association` is a separate object. Keys are sorted when written
so two equivalent reports have a stable textual layout.

## CSV

CSV uses one row per result. `record_type` is either `strategy` or
`track-association`, and `name` identifies the scheduler. Columns that do not apply to
a row are empty. Common metadata is repeated on every row so filtered files remain
self-describing.

## Compatibility

Readers should reject an unsupported `schema_version` instead of guessing field
meaning. Adding a new optional metric does not require changing existing meanings;
renaming or redefining a field requires a new schema version.
