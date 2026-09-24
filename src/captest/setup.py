# src/captest/setup.py
"""Capacity-test setup documents.

A setup is pure data: yaml or json made of strings, numbers, booleans,
lists and tagged mappings. This module holds the typed view of that
document (``TestSetup``), the node grammar (``Group`` / ``Column`` /
``Calc`` / literal), normalisation and identity (``to_dict`` /
``content_digest``), and the validation that needs no data (tier 1 here,
tier 2 in :func:`check_project_fit`). It never imports ``capdata`` or
``captest``; a ``CapData`` reaches it only as a runtime argument.
"""

import difflib
import hashlib
import importlib.util
import inspect
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import yaml
from patsy import ModelDesc, PatsyError
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

from captest.calcparams import CALC_REGISTRY, DOWNSTREAM_PARAMS  # noqa: F401

# --- formula ---------------------------------------------------------------


def parse_regression_formula(formula: str) -> tuple[list[str], list[str]]:
    """
    Return (lhs_list, rhs_list) for `formula`.

    Rules
    -----
    • Each list contains the **unique raw variable names** appearing on
      that side, sorted.
    • `- 1` (intercept-removal) is ignored.
    • `I(...)` blocks are unwrapped; products like `I(poa * t_amb)` are
      split into their component symbols (`poa`, `t_amb`).

    Parameters
    ----------
    formula : str
        Regression formula to parse.

    Returns
    -------
    Tuple[List[str], List[str]]
        Tuple of (lhs_list, rhs_list).
    """
    # --- helpers ------------------------------------------------------
    _sym_re = re.compile(r"[A-Za-z_]\w*")

    def _extract_raw_names(factor_str: str) -> list[str]:
        """
        Turn 'I(poa * t_amb)'  ->  ['poa', 't_amb']
             'poa'             ->  ['poa']
        """
        # strip outer I(…)
        if factor_str.startswith("I(") and factor_str.endswith(")"):
            factor_str = factor_str[2:-1]
        # split by * or :  (products/interactions)
        parts = re.split(r"[\*\:]", factor_str)
        names = []
        for part in parts:
            # pull out identifier tokens
            names.extend(_sym_re.findall(part))
        return names

    # --- main logic ---------------------------------------------------
    md = ModelDesc.from_formula(formula)

    lhs_list: list[str] = []
    rhs_list: list[str] = []

    # left
    for term in md.lhs_termlist:
        for f in term.factors:
            for name in _extract_raw_names(f.name()):
                if name not in lhs_list:
                    lhs_list.append(name)

    # right
    for term in md.rhs_termlist:
        for f in term.factors:
            for name in _extract_raw_names(f.name()):
                if name not in rhs_list:
                    rhs_list.append(name)

    # discard the Patsy-built-in intercept symbol if present
    rhs_list = [n for n in rhs_list if n != "Intercept"]

    return lhs_list, rhs_list


# --- canonical json --------------------------------------------------------


def _validate_canonical(obj, path):
    if obj is None or isinstance(obj, (bool, int, str)):
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"{path}: non-finite float cannot be canonical JSON")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: canonical JSON requires string keys")
            _validate_canonical(value, f"{path}.{key}")
        return
    if isinstance(obj, list):
        for index, value in enumerate(obj):
            _validate_canonical(value, f"{path}[{index}]")
        return
    raise TypeError(f"{type(obj).__name__} at {path} is not JSON-serializable")


def canonical_json(obj):
    """Serialize ``obj`` to the one canonical JSON form used for identity.

    Sorted keys, no insignificant whitespace, non-ASCII kept as-is, NaN and
    Infinity rejected — byte-identical to the form the pft-mono ``ctsweep``
    store hashes.

    Parameters
    ----------
    obj : dict, list or scalar
        Plain JSON-representable data.

    Returns
    -------
    str

    Raises
    ------
    TypeError
        On a value or mapping key that cannot be represented.
    ValueError
        On a non-finite float.
    """
    _validate_canonical(obj, "$")
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


# --- nodes -----------------------------------------------------------------

NODE_TAGS = ("group", "column", "calc")
AGG_FUNCS = ("mean", "sum", "median", "min", "max")
_REP_FUNC_RE = re.compile(r"^(mean|median|perc_\d+)$")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Group(_Frozen):
    """Aggregate a column group; writes ``<group>_<agg>_agg``."""

    group: str
    agg: Literal["mean", "sum", "median", "min", "max"] = "mean"


class Column(_Frozen):
    """A raw column of ``data`` by name."""

    column: str


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Scalar = None | bool | int | FiniteFloat | str
Literal_ = Scalar | list[Scalar]


def _node_kind(node):
    """Name the union member ``node`` belongs to, for the discriminator.

    A mapping is identified by which one of ``NODE_TAGS`` it carries; a
    mapping with none or several is unclassifiable and returns ``None`` so
    pydantic reports ``union_tag_not_found`` at that path. An already-built
    model reports its own class; anything else is a literal.
    """
    if isinstance(node, dict):
        present_tags = [tag for tag in NODE_TAGS if tag in node]
        return present_tags[0] if len(present_tags) == 1 else None
    if isinstance(node, BaseModel):
        return type(node).__name__.lower()
    return "literal"


class Calc(_Frozen):
    """A registered calculation; writes a column named for the registry key."""

    calc: str
    args: dict[str, "Node"] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_against_registry(self):
        entry = CALC_REGISTRY.get(self.calc)
        if entry is None:
            hint = difflib.get_close_matches(self.calc, CALC_REGISTRY, n=1)
            suffix = f" Did you mean {hint[0]!r}?" if hint else ""
            raise ValueError(f"unknown calculation {self.calc!r}.{suffix}")
        params = inspect.signature(entry.func).parameters
        if "output" in params:
            raise ValueError(
                f"calculation {self.calc!r} has a parameter named 'output', which "
                "is reserved by CapData.custom_param; rename the parameter."
            )
        allowed = set(params) - {"data", "verbose"}
        unknown = sorted(set(self.args) - allowed)
        if unknown:
            raise ValueError(
                f"unknown argument(s) {unknown} for {self.calc}; "
                f"expected {sorted(allowed)}"
            )
        required = {
            name
            for name, p in params.items()
            if p.default is p.empty
            and name not in ("data", "verbose")
            and name not in entry.requires_params
        }
        missing = sorted(required - set(self.args))
        if missing:
            raise ValueError(f"missing required argument(s) {missing} for {self.calc}")
        return self


Node = Annotated[
    Annotated[Group, Tag("group")]
    | Annotated[Column, Tag("column")]
    | Annotated[Calc, Tag("calc")]
    | Annotated[Literal_, Tag("literal")],
    Discriminator(_node_kind),
]
Calc.model_rebuild()


def agg_column_name(group, agg):
    """Column a ``Group`` node writes (the ``agg_group`` naming)."""
    return f"{group}_{agg}_agg"


def calc_output_name(node):
    """Column a ``Calc`` node writes: its registry name."""
    return node.calc


def _producers(node, path, out):
    """Collect (output column, path, node) for every producing node."""
    if isinstance(node, dict):
        for key, value in node.items():
            _producers(value, f"{path}.{key}", out)
    elif isinstance(node, Group):
        out.append((agg_column_name(node.group, node.agg), path, node))
    elif isinstance(node, Calc):
        out.append((calc_output_name(node), path, node))
        _producers(node.args, f"{path}.args", out)


def walk_nodes(node, path=""):
    """Yield ``(path, node)`` for every model node in a tree, depth first."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk_nodes(value, f"{path}.{key}" if path else key)
    elif isinstance(node, (Group, Column)):
        yield path, node
    elif isinstance(node, Calc):
        yield path, node
        yield from walk_nodes(node.args, f"{path}.args")


class Side(_Frozen):
    """One side's regression columns: formula variable -> node."""

    reg_cols: dict[str, Node]

    @field_validator("reg_cols", mode="after")
    @classmethod
    def _top_level_nodes_and_unique_outputs(cls, reg_cols):
        for var, node in reg_cols.items():
            if not isinstance(node, (Group, Column, Calc)):
                raise ValueError(
                    f"{var}: a top-level regression column must be a group, "
                    f"column or calc node, not a literal ({node!r})"
                )
        producers = []
        _producers(reg_cols, "reg_cols", producers)
        seen = {}
        for output, path, node in producers:
            if output in seen and seen[output][1] != node:
                raise ValueError(
                    f"two different nodes write column {output!r}: "
                    f"{seen[output][0]} and {path}"
                )
            seen.setdefault(output, (path, node))
        return reg_cols


class RepConditions(_Frozen):
    """Keyword arguments to ``CapData.rep_cond`` in document form."""

    func: dict[str, str] = Field(default_factory=dict)
    w_vel: FiniteFloat | None = None
    irr_bal: bool = False
    percent_filter: FiniteFloat = 20
    front_poa: str = "poa"
    rc_kwargs: dict[str, Scalar] | None = None

    @field_validator("func", mode="after")
    @classmethod
    def _func_values(cls, func):
        bad = {k: v for k, v in func.items() if not _REP_FUNC_RE.match(v)}
        if bad:
            raise ValueError(f"func values must be mean, median or perc_N; got {bad}")
        return func


class TestSetup(_Frozen):
    """A complete capacity-test setup document.

    Parameters
    ----------
    name : str
        Preset or variant name.
    description : str
        Prose description shown by ``captest.test_setups()``.
    derived_from : str or None
        Provenance only (a preset name or content digest); never used to
        fill omitted fields.
    reg_fml : str
        patsy regression formula.
    meas, sim : Side
        Regression columns for the measured and modeled data.
    params : dict
        Test-level parameters this setup requires, checked at tier 2 against
        the effective value each calculation would receive.
    rep_conditions : RepConditions
        Reporting-condition options.
    scatter_plots : str
        Name in ``captest.SCATTER_REGISTRY``; membership is checked by
        ``captest.py``.
    """

    #: Not a pytest test class despite the name pattern: this is imported
    #: into ``tests/test_setup.py`` and would otherwise be collected.
    __test__ = False

    name: str
    description: str = ""
    derived_from: str | None = None
    reg_fml: str
    meas: Side
    sim: Side
    params: dict[str, Scalar] = Field(default_factory=dict)
    rep_conditions: RepConditions = Field(default_factory=RepConditions)
    scatter_plots: str = Field(default="default", min_length=1)

    @field_validator("reg_fml", mode="after")
    @classmethod
    def _formula_parses(cls, reg_fml):
        try:
            parse_regression_formula(reg_fml)
        except PatsyError as exc:
            raise ValueError(f"reg_fml does not parse: {exc}") from exc
        return reg_fml

    @field_validator("meas", "sim", mode="after")
    @classmethod
    def _formula_variables_present(cls, side, info):
        reg_fml = info.data.get("reg_fml")
        if reg_fml is None:
            return side
        lhs, rhs = parse_regression_formula(reg_fml)
        missing = sorted((set(lhs) | set(rhs)) - set(side.reg_cols))
        if missing:
            raise ValueError(f"reg_cols is missing formula variable(s) {missing}")
        return side

    @field_validator("params", mode="after")
    @classmethod
    def _params_are_downstream(cls, params):
        bad = sorted(set(params) - set(DOWNSTREAM_PARAMS))
        if bad:
            raise ValueError(f"params keys {bad} are not CapTest downstream params")
        return params

    @field_validator("rep_conditions", mode="after")
    @classmethod
    def _func_keys_are_rhs(cls, rc, info):
        reg_fml = info.data.get("reg_fml")
        if reg_fml is None:
            return rc
        _, rhs = parse_regression_formula(reg_fml)
        bad = sorted(set(rc.func) - set(rhs))
        if bad:
            raise ValueError(f"func keys {bad} are not rhs variables of reg_fml")
        return rc

    # --- serialization -------------------------------------------------

    def to_dict(self):
        """The normalised document: every field, defaults materialised."""
        return self.model_dump(mode="json")

    def content_digest(self):
        """sha256 of ``canonical_json(self.to_dict())``; the setup's identity."""
        return hashlib.sha256(
            canonical_json(self.to_dict()).encode("utf-8")
        ).hexdigest()

    @classmethod
    def load(cls, source):
        """Build a setup from a yaml/json file path or a mapping.

        Always validates through the model; yaml is read with ``safe_load``
        (json is valid yaml). Use :meth:`loads` for document text.
        """
        if isinstance(source, dict):
            return cls.model_validate(source)
        return cls.loads(Path(source).read_text())

    @classmethod
    def loads(cls, text):
        """Build a setup from yaml or json document text."""
        return cls.model_validate(yaml.safe_load(text))

    def to_yaml(self, path):
        """Write ``to_dict()`` as yaml."""
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def to_json(self, path):
        """Write ``to_dict()`` as json."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def json_schema(cls):
        """JSON Schema of the document shape (an authoring contract)."""
        return cls.model_json_schema()

    @staticmethod
    def derive(base, **changes):
        """Return a new complete setup derived from ``base``.

        A thin delegate to the module-level :func:`captest.setup.derive`,
        which documents the full parameter list, the key-by-key
        ``reg_cols_meas`` / ``reg_cols_sim`` merge, and the ``rep_conditions``
        pruning behaviour.

        Parameters
        ----------
        base : TestSetup
        **changes
            Keyword arguments forwarded to :func:`captest.setup.derive`
            (``name``, ``description``, ``reg_fml``, ``reg_cols_meas``,
            ``reg_cols_sim``, ``params``, ``rep_conditions``,
            ``scatter_plots``).

        Returns
        -------
        TestSetup

        Raises
        ------
        DerivationError
            See :func:`captest.setup.derive`.
        pydantic.ValidationError
            If the result is not a valid setup.
        """
        return derive(base, **changes)


# --- derive ------------------------------------------------------------


class DerivationError(ValueError):
    """A ``reg_cols`` override names a term the derivation cannot apply."""


def _node_to_data(node):
    return node.model_dump(mode="json") if isinstance(node, BaseModel) else node


def merge_reg_cols(base_side, override, formula_vars):
    """Merge a ``reg_cols`` override onto one side, key by key.

    Parameters
    ----------
    base_side : Side
    override : dict or None
        Formula variable -> node (mapping or model), or ``None`` to remove.
    formula_vars : set of str
        Variables of the resulting formula; every non-``None`` override key
        must be one of them.

    Returns
    -------
    dict
        Plain-data ``reg_cols`` mapping ready for ``Side.model_validate``.

    Raises
    ------
    DerivationError
        For ``None`` on a variable the base lacks, or a key that is not a
        formula variable.
    """
    merged = {var: _node_to_data(node) for var, node in base_side.reg_cols.items()}
    for var, node in (override or {}).items():
        if node is None:
            if var not in merged:
                raise DerivationError(
                    f"cannot remove {var!r}: the base setup has no such term"
                )
            del merged[var]
            continue
        if var not in formula_vars:
            raise DerivationError(
                f"override key {var!r} is not a variable of the regression formula "
                f"{sorted(formula_vars)}"
            )
        merged[var] = _node_to_data(node)
    return merged


def derive(
    base,
    *,
    name=None,
    description=None,
    reg_fml=None,
    reg_cols_meas=None,
    reg_cols_sim=None,
    params=None,
    rep_conditions=None,
    scatter_plots=None,
):
    """Return a new complete setup derived from ``base``.

    ``reg_cols_meas`` / ``reg_cols_sim`` merge key by key (a present key
    replaces that variable's whole node, ``None`` removes the variable);
    every other argument replaces its field wholesale. After the formula and
    sides are resolved, ``rep_conditions.func`` entries for variables no
    longer on the right-hand side are pruned. ``derived_from`` is set to
    ``base.name``; ``name`` defaults to ``base.name``.

    Parameters
    ----------
    base : TestSetup
    name, description, reg_fml, scatter_plots : str or None
    reg_cols_meas, reg_cols_sim : dict or None
    params : dict or None
    rep_conditions : dict or RepConditions or None
        Replaces the base's reporting conditions wholesale; callers that
        want a partial merge do it before calling.

    Returns
    -------
    TestSetup

    Raises
    ------
    DerivationError
        See :func:`merge_reg_cols`.
    pydantic.ValidationError
        If the result is not a valid setup.
    """
    data = base.to_dict()
    data["derived_from"] = base.name
    if name is not None:
        data["name"] = name
    if description is not None:
        data["description"] = description
    if reg_fml is not None:
        data["reg_fml"] = reg_fml
    if params is not None:
        data["params"] = dict(params)
    if scatter_plots is not None:
        data["scatter_plots"] = scatter_plots
    if rep_conditions is not None:
        data["rep_conditions"] = _node_to_data(rep_conditions)

    lhs, rhs = parse_regression_formula(data["reg_fml"])
    formula_vars = set(lhs) | set(rhs)
    data["meas"] = {"reg_cols": merge_reg_cols(base.meas, reg_cols_meas, formula_vars)}
    data["sim"] = {"reg_cols": merge_reg_cols(base.sim, reg_cols_sim, formula_vars)}

    func = dict(data["rep_conditions"].get("func") or {})
    data["rep_conditions"]["func"] = {k: v for k, v in func.items() if k in rhs}
    return TestSetup.model_validate(data)


# --- tier 2: project fit ---------------------------------------------------


@dataclass(frozen=True)
class FitError:
    """One tier-2 finding: a document path and what is wrong there."""

    path: str
    message: str


class SetupFitError(ValueError):
    """A setup does not fit the project; ``errors`` lists every finding."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__(
            "setup does not fit this project:\n"
            + "\n".join(f"  {e.path}: {e.message}" for e in self.errors)
        )


def effective_value(name, node, cd):
    """Value a calculation would receive for ``name`` under the precedence rules.

    1. the node's ``args`` entry, if present (an explicit ``None`` counts);
    2. else the ``cd`` attribute of that name when present and not ``None``;
    3. else the function's own default;
    4. else ``inspect.Parameter.empty``.
    """
    if name in node.args:
        return node.args[name]
    attr = getattr(cd, name, None)
    if attr is not None:
        return attr
    param = inspect.signature(CALC_REGISTRY[node.calc].func).parameters.get(name)
    if param is None:
        return inspect.Parameter.empty
    return param.default


def check_project_fit(setup, side, cd):
    """Tier-2 validation of one side of ``setup`` against a project.

    Needs only ``cd.column_groups``, ``cd.data.columns`` and the attributes
    named by each calculation's ``requires_params``; no data is read.

    Parameters
    ----------
    setup : TestSetup
    side : {"meas", "sim"}
    cd : CapData-like

    Returns
    -------
    list of FitError
        Empty when the side fits. Every finding is collected so a document
        can be fixed in one pass.
    """
    errors = []
    reg_cols = getattr(setup, side).reg_cols
    groups = dict(cd.column_groups)
    sensor_columns = {c for cols in groups.values() for c in cols}
    data_columns = set(cd.data.columns)

    for path, node in walk_nodes(reg_cols, f"{side}.reg_cols"):
        if isinstance(node, Group):
            if node.group not in groups:
                errors.append(
                    FitError(f"{path}.group", f"{node.group!r} is not a column group")
                )
            continue
        if isinstance(node, Column):
            if node.column not in data_columns:
                errors.append(
                    FitError(f"{path}.column", f"{node.column!r} is not a column")
                )
            continue
        entry = CALC_REGISTRY[node.calc]
        for package in entry.requires_import:
            if importlib.util.find_spec(package) is None:
                errors.append(
                    FitError(path, f"{node.calc} requires the {package!r} package")
                )
        for name in entry.requires_params:
            value = effective_value(name, node, cd)
            if value is None or value is inspect.Parameter.empty:
                errors.append(
                    FitError(
                        path,
                        f"{node.calc} requires {name}, which resolves to None; set "
                        f"it on the test or pass it in args",
                    )
                )
        output = calc_output_name(node)
        if output in groups or output in sensor_columns:
            errors.append(
                FitError(
                    path,
                    f"output column {output!r} shadows a column group "
                    f"id or sensor column",
                )
            )
        for key, required in setup.params.items():
            if key in entry.requires_params:
                value = effective_value(key, node, cd)
                if value != required:
                    errors.append(
                        FitError(
                            f"params.{key}",
                            f"setup requires {key}={required!r} but {node.calc} at "
                            f"{path} would receive {value!r}",
                        )
                    )
    return errors
