"""Prep steps: declared, serializable adjustments applied between load and setup.

Each step is a ``param``-configured class that rewrites ``CapData.data`` in
place and records itself on ``CapData.prep``, so the adjustment round-trips
through the yaml config and replays after every load.

This module follows ``filters.py``'s import rule: it is imported one-way by
``capdata.py`` and never imports ``capdata``. A step touches a ``CapData``
only through the runtime ``capdata`` argument.
"""

import difflib
import re

import pandas as pd
import param

from captest import util


class BasePrepStep(util.StrictAttrs, param.Parameterized):
    """Common ancestor for data-preparation steps.

    Holds the shared lifecycle (``run``), the three-way column selector, and
    the ``args_repr`` / ``explanation`` rendering used by ``describe_prep``.
    Subclasses implement ``_execute(capdata, columns)``, which mutates
    ``capdata.data`` in place and returns nothing.

    Every step is atomic: ``run`` snapshots ``capdata.data`` and restores it
    if ``_execute`` raises, and appends the step only on success. A failed
    step is therefore a no-op in both ``data`` and ``prep``.

    Runtime state (``columns_resolved``) is set by ``run`` as a plain
    attribute and is never serialized. Attribute assignment is restricted by
    ``util.StrictAttrs``.
    """

    custom_name = param.String(
        default=None,
        allow_None=True,
        doc="Optional display name in the prep description.",
    )
    columns = param.List(
        default=None,
        allow_None=True,
        doc="Explicit column names. Mutually exclusive with group/group_regex.",
    )
    group = param.ClassSelector(
        class_=(str, list),
        default=None,
        allow_None=True,
        doc="column_groups id, or list of ids. Mutually exclusive with the others.",
    )
    group_regex = param.String(
        default=None,
        allow_None=True,
        doc="Regex matched against column_groups ids. Mutually exclusive with others.",
    )

    # Class-intrinsic human-readable template; set by concrete subclasses.
    _explanation_template = None

    # Steps that rewrite values are refused once filters are applied; steps
    # that only change column identity (drop, rename) are not.
    _mutates_values = True

    _runtime_attrs = frozenset({"columns_resolved"})

    def run(self, capdata):
        """Execute the step against ``capdata`` and record it on ``prep``.

        Parameters
        ----------
        capdata : CapData
            Target dataset. ``data`` is mutated in place.

        Returns
        -------
        BasePrepStep
            ``self``, so callers can keep a handle on the recorded step.

        Raises
        ------
        RuntimeError
            If this step rewrites values and ``capdata.filters`` is non-empty.
        ValueError
            If the column selector is not exactly one of ``columns`` /
            ``group`` / ``group_regex``, or resolves to no columns.
        """
        self._check_not_filtered(capdata)
        cols = self._resolve_columns(capdata)
        self.columns_resolved = cols
        snapshot = capdata.data.copy()
        try:
            self._execute(capdata, cols)
        except Exception:
            # A failed step is a no-op: restore data, record nothing.
            capdata.data = snapshot
            raise
        # Reassign rather than append so param watchers fire.
        capdata.prep = capdata.prep + [self]
        return self

    def _check_not_filtered(self, capdata):
        """Raise when a value-rewriting step runs after filtering.

        A filter applied earlier was evaluated against un-prepped values, so
        letting prep run now produces wrong numbers rather than a crash.
        """
        if not self._mutates_values or not capdata.filters:
            return
        raise RuntimeError(
            f"Cannot run {type(self).__name__} after filtering — "
            f"{len(capdata.filters)} filters are already applied and were "
            "evaluated against un-prepped values. Call cd.reset_filter() "
            "first, or reload this side."
        )

    def _resolve_columns(self, capdata):
        """Resolve the selector against ``capdata`` at run time.

        Resolution is deliberately deferred to ``run`` rather than done at
        construction: earlier steps in the same list may have dropped or
        renamed columns.

        Returns
        -------
        list of str
            Column names this step acts on, in ``column_groups`` order for the
            group selectors and in the given order for ``columns``.
        """
        chosen = [
            name
            for name in ("columns", "group", "group_regex")
            if getattr(self, name) is not None
        ]
        if len(chosen) != 1:
            raise ValueError(
                f"{type(self).__name__} requires exactly one of 'columns', "
                f"'group', or 'group_regex'; got {chosen or 'none'}."
            )
        if self.columns is not None:
            cols = list(self.columns)
        else:
            cols = self._columns_from_groups(capdata)
        missing = [c for c in cols if c not in capdata.data.columns]
        if missing:
            raise ValueError(
                f"{type(self).__name__} selector resolved to columns absent "
                f"from data: {missing}."
            )
        if not cols:
            raise ValueError(
                f"{type(self).__name__} selector resolved to zero columns; "
                "a prep step that does nothing is always a mistake."
            )
        return cols

    def _columns_from_groups(self, capdata):
        """Expand the ``group`` / ``group_regex`` selector to column names."""
        groups = capdata.column_groups
        if self.group is not None:
            ids = [self.group] if isinstance(self.group, str) else list(self.group)
            for group_id in ids:
                if group_id not in groups:
                    suggestion = difflib.get_close_matches(group_id, list(groups), n=1)
                    hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
                    raise ValueError(
                        f"Column group {group_id!r} is not in column_groups. "
                        f"Available ids: {sorted(groups)}.{hint}"
                    )
        else:
            pattern = re.compile(self.group_regex)
            ids = [group_id for group_id in groups if pattern.search(group_id)]
            if not ids:
                raise ValueError(
                    f"group_regex {self.group_regex!r} matched no column group. "
                    f"Available ids: {sorted(groups)}."
                )
        cols = []
        for group_id in ids:
            for col in groups[group_id]:
                if col not in cols:
                    cols.append(col)
        return cols

    def _execute(self, capdata, columns):
        """Mutate ``capdata.data`` in place. Implemented by subclasses."""
        raise NotImplementedError

    @staticmethod
    def _require_numeric(capdata, columns, step_name):
        """Raise ``TypeError`` naming the first non-numeric column."""
        for col in columns:
            if not pd.api.types.is_numeric_dtype(capdata.data[col]):
                raise TypeError(
                    f"{step_name} requires numeric columns; {col!r} has dtype "
                    f"{capdata.data[col].dtype}."
                )

    @property
    def args_repr(self):
        """Render the step's params for ``describe_prep``."""
        skip = {"custom_name", "name"}
        items = [
            f"{k}={v}"
            for k, v in util.params_to_config(self).items()
            if k not in skip and v is not None
        ]
        return ", ".join(items) if items else "Default arguments"

    @property
    def explanation(self):
        """Human-readable description of the step's effect (read after run)."""
        if self._explanation_template is None:
            return None
        try:
            values = self._explanation_values()
        except AttributeError:
            return None
        return self._explanation_template.format(**values)

    def _explanation_values(self):
        """Substitution mapping for ``_explanation_template``."""
        return {"columns": ", ".join(self.columns_resolved)}

    def to_config(self):
        """Serialize this step to a yaml-safe config dict."""
        config = {"type": type(self).__name__}
        config.update(util.params_to_config(self))
        return config

    @classmethod
    def from_config(cls, config):
        """Build an instance from a :meth:`to_config` dict."""
        config = {k: v for k, v in config.items() if k != "type"}
        return cls(**config)


class Scale(BasePrepStep):
    """Apply an affine transform ``value * factor + offset`` in place.

    The raw-numbers escape hatch for rescaling that no unit pair covers.
    Writes back to the same column names, which is the property
    ``calcparams.custom_param`` structurally cannot provide.
    """

    factor = param.Number(default=1.0, doc="Multiplier applied to each value.")
    offset = param.Number(default=0.0, doc="Added after multiplying.")

    _explanation_template = (
        "Columns {columns} were scaled by {factor} with an offset of {offset}."
    )

    def _execute(self, capdata, columns):
        self._require_numeric(capdata, columns, type(self).__name__)
        capdata.data[columns] = capdata.data[columns] * self.factor + self.offset

    def _explanation_values(self):
        return {
            "columns": ", ".join(self.columns_resolved),
            "factor": self.factor,
            "offset": self.offset,
        }


UNIT_ALIASES = {
    "degf": "F",
    "fahrenheit": "F",
    "f": "F",
    "degc": "C",
    "celsius": "C",
    "c": "C",
    "k": "K",
    "kelvin": "K",
    "mph": "mph",
    "mi/h": "mph",
    "m/s": "m/s",
    "mps": "m/s",
    "km/h": "km/h",
    "kph": "km/h",
    "kn": "kn",
    "knots": "kn",
    "ft/s": "ft/s",
    "w": "W",
    "kw": "kW",
    "mw": "MW",
    "mbar": "mbar",
    "pa": "Pa",
    "m": "m",
    "cm": "cm",
    "in": "in",
    "inches": "in",
}

# out = value * factor + offset
UNIT_CONVERSIONS = {
    ("F", "C"): (5 / 9, -160 / 9),  # (F - 32) * 5/9
    ("K", "C"): (1.0, -273.15),
    ("mph", "m/s"): (0.44704, 0.0),
    ("km/h", "m/s"): (1 / 3.6, 0.0),
    ("kn", "m/s"): (0.514444, 0.0),
    ("ft/s", "m/s"): (0.3048, 0.0),
    ("W", "kW"): (0.001, 0.0),
    ("MW", "kW"): (1000.0, 0.0),
    ("mbar", "Pa"): (100.0, 0.0),
    ("m", "cm"): (100.0, 0.0),
    ("in", "cm"): (2.54, 0.0),
}


def _normalize_unit(units):
    """Map a user-supplied unit string to its canonical spelling."""
    return UNIT_ALIASES.get(str(units).strip().lower(), str(units).strip())


def conversion_factors(from_units, to_units):
    """Return the ``(factor, offset)`` converting ``from_units`` to ``to_units``.

    Conversions are affine: ``out = value * factor + offset``. Only the forward
    direction is tabulated in :data:`UNIT_CONVERSIONS`; the inverse is derived
    as ``out = (value - offset) / factor``, so the table cannot drift out of
    sync with itself. Aliases are normalized case-insensitively before lookup.

    Parameters
    ----------
    from_units, to_units : str
        Unit names or aliases, e.g. ``"degF"`` and ``"C"``.

    Returns
    -------
    tuple of float
        ``(factor, offset)``.

    Raises
    ------
    ValueError
        If either unit is omitted (``None``), the units are identical, or the
        pair is not supported in either direction.
    """
    if from_units is None or to_units is None:
        raise ValueError(
            "from_units and to_units are both required; got "
            f"from_units={from_units!r}, to_units={to_units!r}."
        )
    src = _normalize_unit(from_units)
    dst = _normalize_unit(to_units)
    if src == dst:
        raise ValueError(
            f"from_units and to_units are identical ({src!r}); a conversion "
            "that does nothing is always a mistake."
        )
    if (src, dst) in UNIT_CONVERSIONS:
        return UNIT_CONVERSIONS[(src, dst)]
    if (dst, src) in UNIT_CONVERSIONS:
        factor, offset = UNIT_CONVERSIONS[(dst, src)]
        return 1 / factor, -offset / factor
    supported = sorted(f"{a} -> {b}" for a, b in UNIT_CONVERSIONS)
    known_units = {u for pair in UNIT_CONVERSIONS for u in pair}
    suggestion = difflib.get_close_matches(src, known_units, n=1)
    hint = f" Did you mean {suggestion[0]!r} for from_units?" if suggestion else ""
    raise ValueError(
        f"No conversion from {src!r} to {dst!r}. Inverses of the supported "
        f"pairs are derived automatically; supported pairs: {supported}. Use "
        f"Scale(factor=..., offset=...) for anything else.{hint}"
    )


class ConvertUnits(BasePrepStep):
    """Convert the selected columns between units, in place.

    The conversion is affine and applied to the same column names, so a site
    with twelve thermocouples converts in one step. This is an asserted
    transformation, not a checked one: nothing tracks or validates a column's
    current units.
    """

    from_units = param.String(default=None, allow_None=True, doc="Source units.")
    to_units = param.String(default=None, allow_None=True, doc="Target units.")

    _explanation_template = (
        "Columns {columns} were converted from {from_units} to {to_units}."
    )

    def _execute(self, capdata, columns):
        self._require_numeric(capdata, columns, type(self).__name__)
        factor, offset = conversion_factors(self.from_units, self.to_units)
        capdata.data[columns] = capdata.data[columns] * factor + offset

    def _explanation_values(self):
        return {
            "columns": ", ".join(self.columns_resolved),
            "from_units": self.from_units,
            "to_units": self.to_units,
        }


class AsType(BasePrepStep):
    """Cast the selected columns to a dtype, in place.

    Covers the PVsyst frames that load as ``object`` because of a stray index
    column and need ``astype(float)`` before any numeric filtering.
    """

    dtype = param.String(default="float64", doc="Target numpy/pandas dtype name.")

    _explanation_template = "Columns {columns} were cast to {dtype}."

    def _execute(self, capdata, columns):
        capdata.data[columns] = capdata.data[columns].astype(self.dtype)

    def _explanation_values(self):
        return {"columns": ", ".join(self.columns_resolved), "dtype": self.dtype}


class DropColumns(BasePrepStep):
    """Drop the selected columns from ``data`` and ``column_groups``.

    Delegates to :meth:`CapData.drop_cols`; the step exists so the action is
    recorded and replayed, not to reimplement it. Changes column identity
    rather than values, so it stays legal after filters are applied.
    """

    _explanation_template = "Columns {columns} were dropped."
    _mutates_values = False

    def _execute(self, capdata, columns):
        capdata.drop_cols(columns, record=False)


class RenameColumns(BasePrepStep):
    """Rename columns in ``data`` and ``column_groups``.

    Takes its columns from ``column_map`` rather than the three-way selector;
    passing a selector is an error. Delegates to :meth:`CapData.rename_cols`.
    """

    column_map = param.Dict(
        default=None, allow_None=True, doc="Mapping of old column name to new."
    )

    _explanation_template = "Columns {columns} were renamed."
    _mutates_values = False

    def _resolve_columns(self, capdata):
        """Return the map's keys; reject the inherited selectors."""
        selectors = [
            name
            for name in ("columns", "group", "group_regex")
            if getattr(self, name) is not None
        ]
        if selectors:
            raise ValueError(
                "RenameColumns takes its columns from 'column_map'; remove "
                f"{selectors}."
            )
        if not self.column_map:
            raise ValueError("RenameColumns requires a non-empty 'column_map'.")
        cols = list(self.column_map)
        missing = [c for c in cols if c not in capdata.data.columns]
        if missing:
            raise ValueError(f"RenameColumns keys are absent from data: {missing}.")
        return cols

    def _execute(self, capdata, columns):
        capdata.rename_cols(self.column_map, record=False)


class Custom(BasePrepStep):
    """Apply an arbitrary callable to the ``CapData`` as a prep step.

    ``func`` is called as ``func(capdata, *args, **kwargs)`` and mutates
    ``capdata.data`` in place; its return value is ignored. This is the escape
    hatch for adjustments the declarative vocabulary does not cover, such as
    blanking east-facing POA columns before sunrise.

    Like ``filters.Custom``, ``func``/``args``/``kwargs`` are plain instance
    attributes rather than ``param`` parameters, and ``func`` serializes to a
    module-qualified name — a lambda cannot be exported.

    The three-way column selector is optional here; when omitted,
    ``columns_resolved`` is an empty list and ``func`` decides what it touches.
    """

    _explanation_template = "Custom prep {call} was applied."
    _runtime_attrs = frozenset({"func", "args", "kwargs"})

    def __init__(self, func, *args, custom_name=None, **kwargs):
        super().__init__(custom_name=custom_name)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def _resolve_columns(self, capdata):
        """Resolve the selector when given; otherwise return no columns."""
        given = [
            name
            for name in ("columns", "group", "group_regex")
            if getattr(self, name) is not None
        ]
        if not given:
            return []
        return super()._resolve_columns(capdata)

    def _execute(self, capdata, columns):
        self.func(capdata, *self.args, **self.kwargs)

    @property
    def args_repr(self):
        """Render ``func_name(arg, ..., k=v, ...)``."""
        name = getattr(self.func, "__name__", repr(self.func))
        arg_parts = [repr(a) for a in self.args]
        kwarg_parts = [f"{k}={v!r}" for k, v in self.kwargs.items()]
        return f"{name}({', '.join(arg_parts + kwarg_parts)})"

    def _explanation_values(self):
        return {"call": self.args_repr}

    def to_config(self):
        return {
            "type": "Custom",
            "func": util.callable_to_qualname(self.func),
            "args": list(self.args),
            "kwargs": dict(self.kwargs),
            "custom_name": self.custom_name,
        }

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        func = util.callable_from_qualname(config["func"])
        args = config.get("args") or []
        kwargs = config.get("kwargs") or {}
        return cls(func, *args, custom_name=config.get("custom_name"), **kwargs)


PREP_REGISTRY = {
    "ConvertUnits": ConvertUnits,
    "Scale": Scale,
    "AsType": AsType,
    "DropColumns": DropColumns,
    "RenameColumns": RenameColumns,
    "Custom": Custom,
}


def prep_step_from_config(d):
    """Build a prep step from a ``to_config()`` dict via ``PREP_REGISTRY``.

    Parameters
    ----------
    d : dict
        Config dict with a ``type`` key naming a registered step.

    Returns
    -------
    BasePrepStep

    Raises
    ------
    ValueError
        If ``type`` is not a registered prep step.
    """
    d = dict(d)
    cls_name = d.pop("type")
    if cls_name not in PREP_REGISTRY:
        suggestion = difflib.get_close_matches(cls_name, PREP_REGISTRY, n=1)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ValueError(f"Unknown prep type {cls_name!r} in prep config.{hint}")
    return PREP_REGISTRY[cls_name].from_config(d)
