import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spectra_scheduler.emitters import (
    BurstEmitter,
    FrequencyHoppingEmitter,
    JitteredPeriodicEmitter,
    ModeSwitchingEmitter,
    PeriodicEmitter,
    ScanningEmitter,
    WindowedEmitter,
)
from spectra_scheduler.receiver import Receiver
from spectra_scheduler.simulation import Simulation


def load_scenario_file(path: str | Path, seed: int = 0) -> Simulation:
    return build_scenario_from_definition(load_scenario_definition(path), seed)


def load_scenario_definition(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read scenario file {source}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("scenario file root must be an object")
    return data


def build_scenario_from_definition(
    definition: Mapping[str, Any],
    seed: int = 0,
) -> Simulation:
    allowed_fields = {"name", "num_bands", "duration", "receiver", "emitters"}
    unknown_fields = set(definition) - allowed_fields
    if unknown_fields:
        unknown = ", ".join(sorted(unknown_fields))
        raise ValueError(f"unknown scenario fields: {unknown}")

    num_bands = _required_integer(definition, "num_bands")
    duration = _required_integer(definition, "duration")
    receiver_data = definition.get("receiver", {})
    if not isinstance(receiver_data, Mapping):
        raise ValueError("receiver must be an object")
    receiver = _build_receiver(receiver_data, seed)

    emitter_data = definition.get("emitters")
    if not isinstance(emitter_data, list):
        raise ValueError("emitters must be an array")
    generator = random.Random(seed)
    emitters = tuple(
        _build_emitter(item, generator, seed, index)
        for index, item in enumerate(emitter_data)
    )
    return Simulation(
        num_bands=num_bands,
        duration=duration,
        emitters=emitters,
        receiver=receiver,
    )


def scenario_name(definition: Mapping[str, Any], fallback: str = "custom") -> str:
    name = definition.get("name", fallback)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("scenario name must be a non-empty string")
    return name.strip()


def _build_receiver(data: Mapping[str, Any], seed: int) -> Receiver:
    values = dict(data)
    seed_offset = values.pop("seed_offset", 0)
    if not isinstance(seed_offset, int):
        raise ValueError("receiver seed_offset must be an integer")
    values["seed"] = seed + seed_offset
    try:
        return Receiver(**values)
    except TypeError as error:
        raise ValueError(f"invalid receiver fields: {error}") from error


def _build_emitter(
    data: object,
    generator: random.Random,
    scenario_seed: int,
    emitter_index: int,
) -> object:
    if not isinstance(data, Mapping):
        raise ValueError("each emitter must be an object")
    values = dict(data)
    emitter_type = values.pop("type", None)
    if not isinstance(emitter_type, str):
        raise ValueError("each emitter requires a string type")

    if emitter_type == "windowed":
        nested = values.pop("emitter", None)
        emitter = _build_emitter(nested, generator, scenario_seed, emitter_index)
        return _construct(WindowedEmitter, {"emitter": emitter, **values}, emitter_type)
    if emitter_type == "mode-switching":
        first_data = values.pop("first_mode", None)
        second_data = values.pop("second_mode", None)
        first_mode = _build_emitter(
            first_data,
            generator,
            scenario_seed,
            emitter_index,
        )
        second_mode = _build_emitter(
            second_data,
            generator,
            scenario_seed,
            emitter_index + 1,
        )
        return _construct(
            ModeSwitchingEmitter,
            {
                "first_mode": first_mode,
                "second_mode": second_mode,
                **values,
            },
            emitter_type,
        )

    constructors = {
        "periodic": (PeriodicEmitter, "period"),
        "frequency-hopping": (FrequencyHoppingEmitter, "period"),
        "scanning": (ScanningEmitter, "period"),
        "burst": (BurstEmitter, "burst_period"),
        "jittered-periodic": (JitteredPeriodicEmitter, "period"),
    }
    try:
        constructor, period_field = constructors[emitter_type]
    except KeyError as error:
        supported = ", ".join((*constructors, "windowed", "mode-switching"))
        raise ValueError(
            f"unknown emitter type {emitter_type!r}; choose from {supported}"
        ) from error

    _resolve_phase(values, period_field, generator)
    if emitter_type == "frequency-hopping" and isinstance(values.get("bands"), list):
        values["bands"] = tuple(values["bands"])
    if emitter_type == "jittered-periodic":
        seed_offset = values.pop("seed_offset", emitter_index + 1)
        if not isinstance(seed_offset, int):
            raise ValueError("jittered emitter seed_offset must be an integer")
        values["seed"] = scenario_seed + seed_offset
    return _construct(constructor, values, emitter_type)


def _resolve_phase(
    values: dict[str, Any],
    period_field: str,
    generator: random.Random,
) -> None:
    phase = values.get("phase", 0)
    if phase != "random":
        return
    period = values.get(period_field)
    if not isinstance(period, int) or period <= 0:
        raise ValueError(f"random phase requires a positive {period_field}")
    values["phase"] = generator.randrange(period)


def _construct(constructor: object, values: dict[str, Any], emitter_type: str) -> object:
    try:
        return constructor(**values)
    except TypeError as error:
        raise ValueError(f"invalid {emitter_type} emitter fields: {error}") from error


def _required_integer(definition: Mapping[str, Any], field_name: str) -> int:
    value = definition.get(field_name)
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value
