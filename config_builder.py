# -*- coding: utf-8 -*-
"""
Scene-file reader.

Two surface syntaxes, one meaning:

  * the keyword-structured ("card") format, which every scene in data/scenes now
    uses -- `Domain`, `Time`, `Solver`, `Load`, `Diffusion`, `Shifting`,
    `Hourglass`, `VelocityLimit`, `Output`, each owning its own options, beside the
    `Materials` list that holds every physical constant; and
  * a flat `Configuration` block of canonical parameter names, which is applied
    ON TOP of the cards as an override layer.

The second exists for two reasons.  It is the legacy format, so an old scene file
still loads unchanged.  And it is what the tests and the tools/ scripts patch --
`cfg.setdefault("Configuration", {}).update(overrides)` on a loaded scene, and
`make_system(..., PST_MA_FIXED=0.05)` in the testkit -- so a card layout never has
to be reproduced by a caller that only wants to change one number.

Everything downstream still reads flat canonical names through `get_cfg()`.  The
card layout stops here.

`params.py` declares what is legal.  Anything not declared there is an error with a
"did you mean" rather than a silent default, which is what the old reader did with
it: `boundaryHandlingMethod` sat in 21 scene files and `PST_R_T` / `PST_N_T` in two,
none of them read by anything.
"""

import json
import os
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # pragma: no cover

from params import (PARAMS, BY_NAME, PRESENCE_FLAGS, CARD_ORDER, SUBCARDS,
                    MUTUALLY_EXCLUSIVE, PASSTHROUGH_SECTIONS, BLOCK_KEYS,
                    BLOCK_REQUIRED, SHAPE_KEYS, SHAPE_REQUIRED, CONSTRAINT_KEYS,
                    CONSTRAINT_REQUIRED, MATERIALS, MATERIALS_EOS, MATERIALS_STRENGTH,
                    MATERIALS_PLASTICITY, MATERIALS_DAMAGE, MATERIAL_ENTRY_KEYS, MATERIAL_ENTRY_REQUIRED,
                    MATERIAL_STRENGTH_KEYS, MATERIAL_STRENGTH_REQUIRED,
                    MATERIAL_PLASTICITY_KEYS, MATERIAL_DAMAGE_KEYS, MATERIAL_DAMAGE_TENSION_KEYS,
                    MATERIALS_DAMAGE_TENSION, MATERIAL_EOS_KEYS,
                    MATERIAL_EOS_REQUIRED_JWL, MATERIAL_EOS_REQUIRED_TAIT,
                    MATERIAL_EOS_BRANCH_KEYS, MATERIAL_JWL_FORBIDDEN,
                    MATERIAL_EOS_REQUIRED_HJC, MATERIAL_PLASTICITY_REQUIRED_HJC,
                    MATERIAL_DAMAGE_REQUIRED_HJC, MATERIAL_DAMAGE_REQUIRED_JC,
                    MATERIAL_DAMAGE_JC_ONLY,
                    MOVED_TO_MATERIALS, HAS_MATERIALS,
                    DETONATOR_KEYS, DETONATOR_REQUIRED, suggest, card_fields)
from materials import (Material, MaterialError, EOS_KINDS, EOS_JWL,
                       EOS_MIE_GRUNEISEN, derive_globals)
from material_models import expand_material_preset, has_library_material


class SceneError(ValueError):
    """A scene file that cannot be read as written.

    Raised with every problem found, not just the first: fixing a deck one error per
    run is how a typo in the last card stays hidden behind a typo in the first.
    """


# --------------------------------------------------------------------------- #
#  type coercion
# --------------------------------------------------------------------------- #
def _coerce(param, value, where, errors):
    """Check and normalise one value; append to `errors` and return None on failure.

    A JSON `null` is treated as "not given", so a deck may carry a key explicitly
    unset without having to delete the line.
    """
    if value is None:
        return None
    t = param.type
    try:
        if t == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError
            return float(value)
        if t == "int":
            if isinstance(value, bool):
                raise TypeError
            if isinstance(value, float) and value != int(value):
                raise TypeError
            if not isinstance(value, (int, float)):
                raise TypeError
            return int(value)
        if t == "bool":
            if isinstance(value, bool):
                return value
            if value in (0, 1):
                return bool(value)
            raise TypeError
        if t == "vec3":
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise TypeError
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
                raise TypeError
            return [float(v) for v in value]
        if t == "list":
            if not isinstance(value, (list, tuple)):
                raise TypeError
            return list(value)
        if t == "str":
            if not isinstance(value, str):
                raise TypeError
            if param.choices:
                canon = {c.lower(): c for c in param.choices}
                if value.lower() not in canon:
                    errors.append("%s: %r is not one of %s."
                                  % (where, value, ", ".join(repr(c) for c in param.choices)))
                    return None
                return canon[value.lower()]       # canonical spelling
            return value
        return value
    except (TypeError, ValueError):
        expect = {"float": "a number", "int": "an integer", "bool": "true or false",
                  "vec3": "a list of three numbers", "list": "a list",
                  "str": "a string"}.get(t, t)
        errors.append("%s: expected %s, got %r." % (where, expect, value))
        return None


def strip_json_comments(text: str) -> str:
    """Strip #-prefixed comments from JSON text, allowing line and in-line comments.

    Preserves '#' characters appearing inside quoted string literals.
    """
    lines = []
    for line in text.splitlines():
        in_string = False
        escape = False
        cut_idx = None
        for i, ch in enumerate(line):
            if ch == '\\' and in_string:
                escape = not escape
                continue
            if ch == '"' and not escape:
                in_string = not in_string
            elif ch == '#' and not in_string:
                cut_idx = i
                break
            escape = False
        if cut_idx is not None:
            line = line[:cut_idx]
        lines.append(line)
    return "\n".join(lines)


def load_json_with_comments(source) -> dict:
    """Parse a JSON file or string that may contain #-prefixed comments.

    Accepts a file path (str), a readable text file-like object, or raw JSON text.
    """
    if hasattr(source, "read"):
        text = source.read()
    elif isinstance(source, str):
        if "\n" not in source and os.path.exists(source):
            with open(source, "r", encoding="utf-8") as fh:
                text = fh.read()
        else:
            text = source
    else:
        raise TypeError(f"Expected path, file, or str, got {type(source)}")

    clean_text = strip_json_comments(text)
    return json.loads(clean_text)


# --------------------------------------------------------------------------- #
#  SimConfig
# --------------------------------------------------------------------------- #
class SimConfig(object):

    def __init__(self, scene_file_path, verbose=True):
        self.path = scene_file_path
        ext = os.path.splitext(scene_file_path)[1].lower() if isinstance(scene_file_path, str) else ""
        try:
            if ext == ".toml":
                with open(scene_file_path, "rb") as fh:
                    raw = tomllib.load(fh)
            else:
                with open(scene_file_path, "r", encoding="utf-8") as fh:
                    raw_text = fh.read()
                clean_text = strip_json_comments(raw_text)
                raw = json.loads(clean_text)
        except tomllib.TOMLDecodeError as e:
            raise SceneError(f"{scene_file_path}: invalid TOML: {e}")
        except json.JSONDecodeError as e:
            raise SceneError(f"{scene_file_path}: invalid JSON: {e}")
        except Exception as e:
            raise SceneError(f"{scene_file_path}: failed to load scene: {e}")

        if not isinstance(raw, dict):
            raise SceneError("%s: the scene must be a JSON or TOML table/object." % scene_file_path)

        self.raw = raw
        self.sources = {}          # flat name -> "scene" | "default" | "derived"
        errors = []
        self.warnings = []

        given = self._read_cards(raw, errors)
        given.update(self._read_flat_overrides(raw, errors))
        self._check_sections(raw, errors)

        # The materials come first, because the two scene-wide scalars the scheme
        # still needs -- c0 and density0 -- are DERIVED from them and have to be in
        # the resolved Configuration before anything validates it.
        #
        # `_check_materials` and `materials.Material` overlap on purpose: the first
        # has the better message for a malformed entry, the second is the authority
        # for a deck that reaches it by some other route.  So the second only runs on
        # a section the first found nothing wrong with -- reporting one mistake twice,
        # in two voices, is worse than reporting it once.
        n_before = len(errors)
        material_names = self._check_materials(raw, errors)
        entries = self._resolve_materials(raw, errors)
        default_delta = given.get("DELTA_SPH")
        if default_delta is None:
            default_delta = 0.0
        self.materials = ({} if len(errors) > n_before
                          else self._build_materials(entries, errors,
                                                    default_delta=default_delta))

        resolved = self._resolve(given, raw, errors,
                                 derive_globals(self.materials))
        self._validate(given, resolved, errors)

        has_particles_file = "ParticlesFile" in raw
        self._check_blocks(raw, errors, material_names, has_particles_file, resolved)
        self._check_constraints(raw, errors)
        self._check_detonators(raw, errors)
        self._check_reactive(raw, resolved, errors)

        if errors:
            raise SceneError(
                "%s: %d problem%s in the scene file:\n  - %s"
                % (os.path.basename(scene_file_path), len(errors),
                   "" if len(errors) == 1 else "s", "\n  - ".join(errors)))

        # The marker the material-constant predicates ask about is an implementation
        # detail of the registry, not a parameter: drop it before the dict becomes the
        # thing the solvers read and the run echoes.
        resolved.pop(HAS_MATERIALS, None)

        # Downstream reads flat canonical names, and SOLID.__init__ writes the derived
        # c0 / exponent back through this same dict, so it has to stay mutable and has
        # to be the thing get_cfg() reads.
        self.config = {"Configuration": resolved}
        for section in PASSTHROUGH_SECTIONS:
            if section in raw:
                self.config[section] = raw[section]

        # Store ParticlesFile if present
        if "ParticlesFile" in raw:
            resolved["ParticlesFile"] = raw["ParticlesFile"]

        RED = "\033[1;31m"
        RESET = "\033[0m"
        for w in self.warnings:
            if "WARNING" in w:
                print(f"{RED}{w}{RESET}")
            else:
                print(f"{RED}WARNING: {w}{RESET}")
        if any("Diffusion.pairs is NOT set to 'sameMaterial'" in w for w in self.warnings):
            self._warned_delta_pairs = True
        if verbose:
            print(self.summary())

    # ------------------------------------------------------------------ read --
    def _read_cards(self, raw, errors):
        """Walk the card sections into flat canonical names."""
        given = {}
        known_sections = set(CARD_ORDER) | set(PASSTHROUGH_SECTIONS) | {"Configuration", "ParticlesFile"}

        for section, body in raw.items():
            if section not in known_sections:
                errors.append("unknown top-level section %r.%s"
                              % (section, suggest(section, known_sections)))
                continue
            if section not in CARD_ORDER:
                continue
            if not isinstance(body, dict):
                errors.append("card %s: expected an object, got %r." % (section, body))
                continue
            self._read_one_card(section, (section,), body, given, errors)
        return given

    def _read_one_card(self, label, path, body, given, errors):
        fields = card_fields(path)
        subs = SUBCARDS.get(label, ()) if len(path) == 1 else ()

        for key, value in body.items():
            if key in subs:
                sub = path + (key,)
                if not isinstance(value, dict):
                    errors.append("card %s.%s: expected an object, got %r."
                                  % (label, key, value))
                    continue
                if sub in PRESENCE_FLAGS:
                    given[PRESENCE_FLAGS[sub]] = True
                    self.sources[PRESENCE_FLAGS[sub]] = "scene"
                self._read_one_card("%s.%s" % (label, key), sub, value, given, errors)
                continue

            param = fields.get(key)
            if param is None:
                if key in MOVED_TO_MATERIALS:
                    # The key that used to be on the `Material` card and is now a
                    # property of a named material.  Say where it went: a "did you
                    # mean" cannot help with a key that moved rather than misspelt.
                    errors.append(
                        "card %s: %s is a property of a MATERIAL, not of the solver, "
                        "and belongs in a Materials[] entry as %s.  The %s card now "
                        "carries the constitutive model and its numerical "
                        "regularisers, and not one physical constant."
                        % (label, key, MOVED_TO_MATERIALS[key], label.split(".")[0]))
                    continue
                hint = suggest(key, set(fields) | set(subs), prefix="%s." % label)
                if not hint:
                    # A field of one of this card's sub-cards is the likely mistake:
                    # writing uTarget directly under Hourglass rather than under
                    # Hourglass.adaptive, which is where it stops being inert.
                    for sub in subs:
                        inner = card_fields(path + (sub,))
                        if key in inner:
                            hint = ("  %s belongs in the %s.%s sub-card, whose presence "
                                    "is what switches the feature on."
                                    % (key, label, sub))
                            break
                if not hint and key in BY_NAME:
                    hint = "  %s is the flat name of card %s." % (key, BY_NAME[key].path)
                errors.append("card %s: unknown key %r.%s" % (label, key, hint))
                continue

            where = "%s.%s" % (label, key)
            coerced = _coerce(param, value, where, errors)
            if coerced is not None:
                given[param.name] = coerced
                self.sources[param.name] = "scene"

    def _read_flat_overrides(self, raw, errors):
        """The flat `Configuration` block: canonical names, applied over the cards."""
        flat = raw.get("Configuration")
        if flat is None:
            return {}
        if not isinstance(flat, dict):
            errors.append("Configuration: expected an object, got %r." % flat)
            return {}

        out = {}
        legal = set(BY_NAME) | set(PRESENCE_FLAGS.values())
        for key, value in flat.items():
            if key in PRESENCE_FLAGS.values():
                coerced = value if isinstance(value, bool) else bool(value)
                out[key] = coerced
                self.sources[key] = "scene"
                continue
            param = BY_NAME.get(key)
            if param is None:
                errors.append("Configuration: unknown parameter %r.%s"
                              % (key, suggest(key, legal)))
                continue
            coerced = _coerce(param, value, "Configuration.%s" % key, errors)
            if coerced is not None:
                out[key] = coerced
                self.sources[key] = "scene"
        return out

    def _check_sections(self, raw, errors):
        for section in PASSTHROUGH_SECTIONS:
            if section in raw and not isinstance(raw[section], list):
                hint = ""
                if isinstance(raw[section], dict):
                    hint = f" In TOML, declare array entries with '[[{section}]]' (array-of-tables) rather than '[{section}]'."
                errors.append("%s: expected a list, got %s.%s"
                              % (section, type(raw[section]).__name__, hint))

    # --------------------------------------------------------------- resolve --
    def _resolve(self, given, raw, errors, derived=None):
        """Fill in defaults. Literal defaults first, then the callable ones, which may
        read the values resolved in the first pass (PST_LAMBDA_LO reads PST_MODE;
        PST_CLAMP reads PST_MODE and PST_MA_MODE).

        `derived` holds the scene-wide scalars computed from `Materials[]` -- `c0`,
        `density0`, `exponent`.  They are neither written by the scene nor a registry
        default, so they get their own provenance, and they are applied last: a deck
        that declares its materials properly cannot also state them, which is what the
        `_no_materials` predicate on each of those parameters enforces.
        """
        resolved = {}

        # Asked by the `required` and `applies` predicates of every material constant,
        # which need to know whether this deck declares its materials in Materials[]
        # or states them flat in the legacy way.  Set before anything reads `resolved`.
        resolved[HAS_MATERIALS] = bool(raw.get("Materials"))

        # Presence flags first: the sub-card being there is the switch.
        for sub, flag in PRESENCE_FLAGS.items():
            if flag not in given:
                resolved[flag] = False
                self.sources.setdefault(flag, "default")
            else:
                resolved[flag] = given[flag]

        # PST is the one card whose absence, not just its `enabled` field, turns the
        # feature off: a scene with no Shifting card at all does no shifting.
        shifting_present = "Shifting" in raw

        deferred = []
        for p in PARAMS:
            if p.name in given:
                resolved[p.name] = given[p.name]
                continue
            if callable(p.default):
                deferred.append(p)
                continue
            if p.name == "PST_ENABLED" and not shifting_present:
                resolved[p.name] = False
            else:
                resolved[p.name] = p.default
            self.sources.setdefault(p.name, "default")

        for p in deferred:
            try:
                resolved[p.name] = p.default(resolved)
            except Exception as exc:                       # pragma: no cover
                errors.append("could not resolve the default for %s: %s" % (p.name, exc))
                resolved[p.name] = None
            self.sources.setdefault(p.name, "default")

        for name, value in (derived or {}).items():
            resolved[name] = value
            self.sources[name] = "derived"

        return resolved

    # -------------------------------------------------------------- validate --
    def _validate(self, given, resolved, errors):
        for p in PARAMS:
            required = p.required(resolved) if callable(p.required) else p.required
            if required and resolved.get(p.name) is None:
                errors.append("%s is required%s and is missing."
                              % (p.path, self._why(p, resolved)))

            if p.name in given and p.applies is not None and not p.applies(resolved):
                msg = "%s was given, but it has no effect here." % p.path
                if p.inert_msg:
                    msg += "  " + p.inert_msg
                errors.append(msg)

        for group in MUTUALLY_EXCLUSIVE:
            present = [k for k in group if k in given]
            if len(present) > 1:
                errors.append("give only one of %s -- they are two ways of writing the "
                              "same quantity; %s were both given."
                              % (", ".join(group), " and ".join(present)))

        # The shock viscosity sub-card without a coefficient is q = 0, which is exactly
        # what leaving the card out already gives.
        if self._subcard_present("Solver", "bulkViscosity") and \
                not (resolved.get("BULK_VISCOSITY_Q") or 0.0) > 0.0:
            errors.append("card Solver.bulkViscosity: needs quadratic > 0; at 0 the "
                          "viscosity is off and the card has no effect.")

        # And for Monaghan's pair term.  `epsilon` alone is not a switch: at
        # alpha = beta = 0 there is no Pi_ij for it to regularise.
        if self._subcard_present("Solver", "artificialViscosity") and \
                not ((resolved.get("MONAGHAN_ALPHA") or 0.0) > 0.0 or
                     (resolved.get("MONAGHAN_BETA") or 0.0) > 0.0):
            errors.append("card Solver.artificialViscosity: needs alpha > 0 or "
                          "beta > 0; at both zero there is no Pi_ij and the card has "
                          "no effect.")

        r = resolved.get("tangentModulusRatio")
        if r is not None and not 0.0 <= r < 1.0:
            errors.append("tangentModulusRatio must be in [0, 1): it is E_t/E, and "
                          "E_t >= E is not a hardening law (got %r)." % r)

        if (resolved.get("GAMMA_SPH") or 0.0) < 0.0:
            errors.append("Diffusion.gamma.coefficient must be >= 0, got %r."
                          % resolved["GAMMA_SPH"])

        if resolved.get("axisymmetric") and resolved.get("planeStrain"):
            errors.append("Domain.planeStrain and Domain.axisymmetric are two different "
                          "readings of the same single particle layer -- plane strain "
                          "extrudes it along z, axisymmetry revolves it about x -- so "
                          "only one of them can be true.")

        sym = resolved.get("symmetryPlanes")
        if sym:
            sym = str(sym).strip().lower()
            bad = sorted(set(sym) - set("yz"))
            if bad:
                extra = (" An x = 0 plane is not offered: the mirror code is two bits "
                         "wide and no deck has wanted one." if "x" in bad else "")
                errors.append("Domain.symmetryPlanes must be a subset of \"yz\" (for "
                              "example \"y\" or \"yz\"), got %r; unknown: %s.%s"
                              % (resolved["symmetryPlanes"], ", ".join(repr(c) for c in bad),
                                 extra))
            if len(set(sym)) != len(sym):
                errors.append("Domain.symmetryPlanes repeats a plane (%r); each plane is "
                              "named once." % resolved["symmetryPlanes"])
            if resolved.get("axisymmetric"):
                errors.append("Domain.symmetryPlanes needs a full 3D domain, and "
                              "Domain.axisymmetric is a single layer that already mirrors "
                              "about y = 0 and carries the 1/r sources (11). Only one of "
                              "them can be on.")
            if resolved.get("planeStrain"):
                errors.append("Domain.symmetryPlanes needs a full 3D domain, and "
                              "Domain.planeStrain pins z to a single layer (4), which "
                              "leaves no third dimension for a plane to cut.")

        lo, hi = resolved.get("PST_LAMBDA_LO"), resolved.get("PST_LAMBDA_HI")
        if lo is not None and hi is not None and lo > hi:
            errors.append("Shifting.lambdaLo (%g) must not exceed Shifting.lambdaHi (%g)."
                          % (lo, hi))

        # Two configurations that are wrong in a production scene but legitimate in a
        # test, so they warn rather than fail: t17 drives the velocity clamp under a
        # fixed Mach number on purpose, and most of the constitutive tests run a solid
        # with no hourglass control because they measure the stress update alone.
        if (resolved.get("PST_ENABLED")
                and str(resolved.get("PST_CLAMP") or "").lower() in ("velocity", "both")
                and str(resolved.get("PST_MA_MODE") or "").lower() == "fixed"):
            self.warnings.append(
                "Shifting.clamp %r caps the shift at mDelta*|v_i| while Shifting.maMode "
                "is 'fixed'.  Wherever the material is at rest the shift is clamped to "
                "zero and the regularisation is off -- which is what maMode 'fixed' "
                "exists to avoid (10.1)." % resolved["PST_CLAMP"])

        if resolved.get("simulationMethod") == "hypoElastic":
            # Not for a charge, which is a gas: section 6.4's argument is that the
            # kernel-sum velocity gradient has a nullspace a SOLID carries load
            # through, and a material with no shear stiffness carries none.  The
            # fluid solver has no hourglass control at all for the same reason.  So
            # the warning is owed to a deck that has a material WITH a strength model
            # in it, which for a legacy flat deck means one that is not on the JWL
            # branch.
            if self.materials:
                any_strength = any(m.has_strength for m in self.materials.values())
            else:
                any_strength = str(resolved.get("EOS_TYPE") or "").lower() != "jwl"
            if not (resolved.get("HOURGLASS_ALPHA") or 0.0) > 0.0 and any_strength:
                self.warnings.append(
                    "Hourglass.alpha is 0 for a hypoElastic material.  Hourglass control "
                    "is mandatory in a boundary-value problem: without it the specimen "
                    "carries 15% of the correct load, at the wrong sign for small "
                    "amplitudes (6.4).")
            if not resolved.get("PST_ENABLED"):
                self.warnings.append(
                    "Shifting is off for a hypoElastic material.  PST is mandatory in a "
                    "boundary-value problem: without it the specimen carries essentially "
                    "no load (6.4).")

        if (resolved.get("simulationMethod") == "hypoElastic"
                and str(resolved.get("DELTA_PAIRS") or "").lower() != "samematerial"):
            self.warnings.append(
                ("\n" + "=" * 78 + "\n"
                 "*** WARNING: Diffusion.pairs is NOT set to 'sameMaterial' (currently: %r)! ***\n"
                 "------------------------------------------------------------------------------\n"
                 "  Across multi-material interfaces or density contrasts, ungated delta-SPH\n"
                 "  density diffusion treats physical density jumps (rho_j - rho_i) as numerical\n"
                 "  errors and smooths them. This acts as an artificial mass pump pushing density\n"
                 "  into lighter materials, leading to severe overcompression, non-physical\n"
                 "  pressure spikes, and numerical divergence (NaN).\n"
                 "\n"
                 "  Diffusion.pairs: 'sameMaterial' is the default and strongly recommended setting\n"
                 "  for all multi-material simulations (see CODE_DESCRIPTION.md Section 13.6).\n"
                 + "=" * 78)
                % resolved.get("DELTA_PAIRS"))

    @staticmethod
    def _why(p, resolved):
        if not callable(p.required):
            return ""
        if not resolved.get(HAS_MATERIALS):
            return (" of a deck that states its material constants flat, with no "
                    "Materials[] section to derive them from")
        return " for a %s material" % resolved.get("simulationMethod")

    def _subcard_present(self, card, sub):
        body = self.raw.get(card)
        return isinstance(body, dict) and isinstance(body.get(sub), dict)

    # -------------------------------------------------------------- materials --
    def _check_materials(self, raw, errors):
        """
        Validate the `Materials` list, which is where every material constant lives.

        One entry is a name, a reference density, an `eos` sub-card saying how its
        pressure is computed, and -- if the material is a solid -- a `strength`
        sub-card saying how it carries shear and when it yields.  An entry with no
        strength model is a fluid: `G = 0`, no deviatoric stress, no yield point, and
        its whole response is the equation of state.  That is not a new capability, it
        is what a JWL material has always been.

        Everything the deck sets that is NOT a property of a material -- `viscosity`,
        `allowNegativePressure`, `dampingCoefficient`, Monaghan's pair viscosity, the
        shock viscosity, and every numerical knob in `Hourglass`, `Shifting` and
        `Diffusion` -- stays on the `Solver` card, because those regularise the SCHEME
        rather than describe a material, and splitting them per object would multiply
        the surface for no physical gain.

        Returns the set of declared names, which `_check_blocks` checks each block's
        `material` key against.
        """
        names = set()
        mats = raw.get("Materials")
        if mats is not None and not isinstance(mats, list):
            return names
        expanded = []
        for i, raw_mat in enumerate(mats or []):
            if isinstance(raw_mat, dict):
                lib_name = raw_mat.get("library")
                if lib_name is not None and not has_library_material(str(lib_name)):
                    errors.append(
                        "Materials[%r]: unknown library material %r."
                        % (raw_mat.get("name", i), lib_name))
                try:
                    expanded_mat = expand_material_preset(raw_mat)
                    expanded.append(expanded_mat)
                except Exception as exc:
                    errors.append("Materials[%r]: %s" % (raw_mat.get("name", i), exc))
                    expanded.append(raw_mat)
            else:
                expanded.append(raw_mat)
        if "Materials" in raw and raw["Materials"] is not None:
            raw["Materials"] = expanded

        for i, mat in enumerate(raw.get("Materials", []) or []):
            where = "Materials[%d]" % i
            if not isinstance(mat, dict):
                errors.append("%s: expected an object." % where)
                continue

            name = mat.get("name")
            if not isinstance(name, str) or not name:
                errors.append("%s: name is required and must be a non-empty string." % where)
            elif name in names:
                errors.append("%s: duplicate material name %r." % (where, name))
            else:
                names.add(name)
                where = "Materials[%r]" % name

            for key in mat:
                if key not in MATERIAL_ENTRY_KEYS:
                    errors.append("%s: unknown key %r.%s"
                                  % (where, key, suggest(key, MATERIAL_ENTRY_KEYS)))
            for field in MATERIAL_ENTRY_REQUIRED:
                if field not in mat:
                    errors.append("%s: %s is required." % (where, field))

            eos = mat.get("eos")
            strength = mat.get("strength")
            kind = str((eos or {}).get("type", "")).lower() if isinstance(eos, dict) else ""

            if kind == "jwl":
                for field in MATERIAL_JWL_FORBIDDEN:
                    if field in mat:
                        errors.append(
                            "%s: %s was given alongside eos.type 'jwl', where it has no "
                            "effect.  Detonation products have no shear stiffness and no "
                            "bulk modulus of their own; the whole volumetric response "
                            "comes from the JWL branch and the deviatoric one is "
                            "identically zero." % (where, field))
                strength = None

            damage = mat.get("damage")
            self._check_hjc_triple(eos, strength, damage, where, errors)
            if eos is not None:
                self._check_material_eos(eos, "%s.eos" % where, errors,
                                         mat.get("density0"), strength is not None)
            if strength is not None:
                self._check_material_strength(strength, "%s.strength" % where, errors)
            if damage is not None:
                self._check_material_damage(damage, "%s.damage" % where, errors)
        return names

    @staticmethod
    def _check_hjc_triple(eos, strength, damage, where, errors):
        """Holmquist-Johnson-Cook is one model in three cards, and all three or none.

        The strength surface reads the EOS's tensile strength and the damage, the
        damage law reads the strength's f_c and the EOS's plastic volumetric strain,
        and the EOS's tensile cutoff reads the damage (6.11).  Any one of them without
        the others would read constants that do not exist, so a partial HJC material
        is refused here rather than half-built.
        """
        eos_hjc = isinstance(eos, dict) and str(eos.get("type", "")).lower() == "hjc"
        plast = strength.get("plasticity") if isinstance(strength, dict) else None
        plast_hjc = (isinstance(plast, dict)
                     and str(plast.get("model", "")).lower() == "hjc")
        dam_hjc = (isinstance(damage, dict)
                   and str(damage.get("model", "")).lower() == "hjc")
        if (eos_hjc or plast_hjc or dam_hjc) and not (eos_hjc and plast_hjc and dam_hjc):
            errors.append(
                "%s: the Holmquist-Johnson-Cook model is declared in three cards that read "
                "each other's constants -- eos.type 'hjc', strength.plasticity.model 'hjc' "
                "and damage.model 'hjc' -- and all three must be given together; got "
                "eos %s, plasticity %s, damage %s."
                % (where, "hjc" if eos_hjc else "not hjc",
                   "hjc" if plast_hjc else "not hjc", "hjc" if dam_hjc else "not hjc"))

    @staticmethod
    def _check_material_eos(eos, where, errors, density0=None, has_strength=False):
        """The `eos` sub-card: which expression turns this material's density into a
        pressure, and the constants that expression needs.

        The branches do not share constants, so a key from the wrong one is an error
        rather than a value that silently does nothing -- the same treatment the cards
        give an inert key.  `bulkModulus` is the one key whose legality depends on
        something outside the sub-card: with a strength model K is DERIVED from
        youngsModulus and poissonRatio, and a second, independent statement of it
        could only contradict them.
        """
        if not isinstance(eos, dict):
            errors.append("%s: expected an object." % where)
            return
        for key in eos:
            if key not in MATERIAL_EOS_KEYS:
                errors.append("%s: unknown key %r.%s"
                              % (where, key, suggest(key, MATERIAL_EOS_KEYS)))
        kind = str(eos.get("type", "")).lower()
        if kind not in EOS_KINDS:
            errors.append("%s: type must be one of %s, got %r."
                          % (where, ", ".join("'%s'" % k for k in sorted(EOS_KINDS)),
                             eos.get("type")))
            return

        foreign = sorted((set(eos) - {"type"} - MATERIAL_EOS_BRANCH_KEYS[kind])
                         & set().union(*MATERIAL_EOS_BRANCH_KEYS.values()))
        if foreign:
            errors.append("%s: %s %s to another equation of state and %s nothing "
                          "under type %r."
                          % (where, ", ".join(foreign),
                             "belong" if len(foreign) > 1 else "belongs",
                             "mean" if len(foreign) > 1 else "means", kind))

        if kind == "jwl":
            for field in MATERIAL_EOS_REQUIRED_JWL:
                if field not in eos:
                    errors.append("%s: %s is required under type 'jwl'." % (where, field))
            SimConfig._check_cj_density(eos.get("cjDensity"), density0, where, errors)
            SimConfig._check_omega(eos.get("omega"), where, errors)
        elif kind == "hjc":
            for field in MATERIAL_EOS_REQUIRED_HJC:
                if field not in eos:
                    errors.append("%s: %s is required under type 'hjc'." % (where, field))
        elif kind == "tait":
            for field in MATERIAL_EOS_REQUIRED_TAIT:
                if field not in eos:
                    errors.append("%s: %s is required under type 'tait'." % (where, field))
        elif kind in ("mie_gruneisen", "polynomial_mie_gruneisen", "mg"):
            has_direct = "C1" in eos
            if not has_direct:
                for field in ("c0", "s"):
                    if field not in eos:
                        errors.append("%s: %s is required under type %r." % (where, field, kind))
                if "gamma0" not in eos and "Gamma0" not in eos and "gruneisen" not in eos:
                    errors.append("%s: gamma0 is required under type %r." % (where, kind))
        else:                                                       # linear
            if has_strength and "bulkModulus" in eos:
                errors.append(
                    "%s: bulkModulus was given alongside a strength model, where it is "
                    "DERIVED: K = E/(3(1 - 2nu)) from the strength card's youngsModulus "
                    "and poissonRatio.  Stating it here could only contradict them."
                    % where)
            elif not has_strength and "bulkModulus" not in eos:
                errors.append(
                    "%s: bulkModulus is required under type 'linear' on a material with "
                    "no strength model; there is no youngsModulus or poissonRatio to "
                    "derive K from.  (A strengthless material that wants a sound speed "
                    "rather than a modulus can use type 'tait' with exponent 1, which "
                    "is the identical expression.)" % where)

    @staticmethod
    def _check_material_strength(strength, where, errors):
        """The `strength` sub-card: present means the material is a solid.

        `hypoElastic` is the only model there is, and `type` is written out rather
        than assumed so that a second one would be a new value here and not a new
        sub-card.  (Johnson-Cook turned out to be a plasticity model, not a strength type.)
        """
        if not isinstance(strength, dict):
            errors.append("%s: expected an object." % where)
            return
        for key in strength:
            if key not in MATERIAL_STRENGTH_KEYS:
                errors.append("%s: unknown key %r.%s"
                              % (where, key, suggest(key, MATERIAL_STRENGTH_KEYS)))
        plast = strength.get("plasticity")
        is_hjc = (isinstance(plast, dict)
                  and str(plast.get("model", "")).lower() == "hjc")
        if is_hjc:
            # K is the EOS's P_crush/mu_crush; E and nu beside it could only
            # contradict it, so an HJC strength card states G alone.
            for field in ("type", "shearModulus"):
                if field not in strength:
                    errors.append("%s: %s is required." % (where, field))
            for field in ("youngsModulus", "poissonRatio"):
                if field in strength:
                    errors.append(
                        "%s: %s was given on a Holmquist-Johnson-Cook material, whose bulk "
                        "modulus is K = crushPressure/crushStrain from the eos card; give "
                        "shearModulus alone." % (where, field))
        else:
            for field in MATERIAL_STRENGTH_REQUIRED:
                if field not in strength:
                    errors.append("%s: %s is required." % (where, field))
            if "shearModulus" in strength:
                errors.append("%s: shearModulus is legal only on a Holmquist-Johnson-Cook "
                              "material; elsewhere G is derived from youngsModulus and "
                              "poissonRatio." % where)
        kind = strength.get("type")
        if kind is not None and str(kind).lower() != "hypoelastic":
            errors.append("%s: type must be 'hypoElastic', got %r.  It is the only "
                          "strength model there is." % (where, kind))

        plasticity = strength.get("plasticity")
        if plasticity is None:
            return
        pwhere = "%s.plasticity" % where
        if not isinstance(plasticity, dict):
            errors.append("%s: expected an object." % pwhere)
            return
        for key in plasticity:
            if key not in MATERIAL_PLASTICITY_KEYS:
                errors.append("%s: unknown key %r.%s"
                              % (pwhere, key, suggest(key, MATERIAL_PLASTICITY_KEYS)))
        p_model = str(plasticity.get("model", "linear")).lower()
        if p_model in ("johnson_cook", "hollomon", "power_law"):
            model_label = "Johnson-Cook" if p_model == "johnson_cook" else "Hollomon"
            if plasticity.get("A") is None and plasticity.get("yieldStress") is None:
                errors.append("%s: %s plasticity requires parameter 'A' (or 'yieldStress')."
                              % (pwhere, model_label))
            for num_key in ("A", "B", "n", "C", "eps0_dot", "T0", "Tm", "m", "Cp", "chi"):
                val = plasticity.get(num_key)
                if val is not None:
                    try:
                        fval = float(val)
                        if num_key in ("A", "B", "n", "C", "Tm", "Cp", "chi") and fval < 0.0:
                            errors.append("%s.%s must be non-negative, got %r." % (pwhere, num_key, val))
                        elif num_key == "eps0_dot" and fval <= 0.0:
                            errors.append("%s.%s must be positive, got %r." % (pwhere, num_key, val))
                    except (TypeError, ValueError):
                        pass    # _resolve_materials reports the type error
        elif p_model == "hjc":
            for field in MATERIAL_PLASTICITY_REQUIRED_HJC:
                if plasticity.get(field) is None:
                    errors.append("%s: %s is required under model 'hjc'." % (pwhere, field))
            for field in ("yieldStress", "yieldStrain", "hardeningModulus",
                          "tangentModulusRatio", "n", "T0", "Tm", "m", "Cp", "chi"):
                if field in plasticity:
                    errors.append("%s: %s means nothing under model 'hjc'." % (pwhere, field))
        elif p_model == "linear":
            if plasticity.get("yieldStress") is None and plasticity.get("yieldStrain") is None:
                errors.append("%s: needs yieldStress or yieldStrain; without one, "
                              "plasticity is off and the card has no effect." % pwhere)
            if "yieldStress" in plasticity and "yieldStrain" in plasticity:
                errors.append("%s: give only one of yieldStress, yieldStrain." % pwhere)
            if "hardeningModulus" in plasticity and "tangentModulusRatio" in plasticity:
                errors.append("%s: give only one of hardeningModulus, tangentModulusRatio."
                              % pwhere)
            r = plasticity.get("tangentModulusRatio")
            if r is not None:
                try:
                    if not 0.0 <= float(r) < 1.0:
                        errors.append("%s.tangentModulusRatio must be in [0, 1), got %r."
                                      % (pwhere, r))
                except (TypeError, ValueError):
                    pass    # _resolve_materials reports the type error
        else:
            errors.append("%s: unknown plasticity model %r. Supported: 'linear', 'johnson_cook', 'hollomon', 'power_law', 'hjc'."
                          % (pwhere, plasticity.get("model")))

    @staticmethod
    def _check_material_damage(damage, where, errors):
        """The `damage` sub-card: continuum spallation and damage parameters."""
        if not isinstance(damage, dict):
            errors.append("%s: expected an object." % where)
            return
        for key in damage:
            if key not in MATERIAL_DAMAGE_KEYS:
                errors.append("%s: unknown key %r.%s"
                              % (where, key, suggest(key, MATERIAL_DAMAGE_KEYS)))
        if str(damage.get("model", "")).lower() == "hjc":
            for field in MATERIAL_DAMAGE_REQUIRED_HJC:
                if damage.get(field) is None:
                    errors.append("%s: %s is required under model 'hjc'." % (where, field))
        d_model = str(damage.get("model", "")).lower()
        if d_model in ("johnson_cook", "jc"):
            for field in MATERIAL_DAMAGE_REQUIRED_JC:
                if damage.get(field) is None:
                    errors.append("%s: %s is required under model 'johnson_cook'."
                                  % (where, field))
        else:
            for field in MATERIAL_DAMAGE_JC_ONLY:
                if damage.get(field) is not None:
                    errors.append("%s: %s is read only under model 'johnson_cook'."
                                  % (where, field))
        tension = damage.get("tension")
        if tension is not None:
            twhere = "%s.tension" % where
            if str(damage.get("model", "")).lower() != "hjc":
                errors.append("%s: a tension card is legal only under damage model 'hjc', "
                              "whose tensile cutoff it replaces." % twhere)
            if not isinstance(tension, dict):
                errors.append("%s: expected an object." % twhere)
                return
            for key in tension:
                if key not in MATERIAL_DAMAGE_TENSION_KEYS:
                    errors.append("%s: unknown key %r.%s"
                                  % (twhere, key, suggest(key, MATERIAL_DAMAGE_TENSION_KEYS)))
            for field in ("model", "referenceVolume"):
                if tension.get(field) is None:
                    errors.append("%s: %s is required." % (twhere, field))

    @staticmethod
    def _check_cj_density(rho_cj, rho0, where, errors):
        """V_CJ = rho0/rho_CJ has to be below 1, and the volume burn divides by 1 - it.

        The Chapman-Jouguet state is a COMPRESSED state, so rho_CJ > rho0 always.  The
        two are separate keys with no other relation between them, so nothing else
        would notice them being swapped -- and swapped they give a negative burn
        fraction from the volume half and a division that is only not by zero by luck.
        """
        try:
            rho_cj, rho0 = float(rho_cj), float(rho0)
        except (TypeError, ValueError):
            return
        if not rho_cj > rho0:
            errors.append(
                "%s: cjDensity (%g) must exceed density0 (%g).  The Chapman-Jouguet "
                "state is a compressed state, so V_CJ = density0/cjDensity is below 1 "
                "-- and the volume half of the burn divides by 1 - V_CJ."
                % (where, rho_cj, rho0))

    @staticmethod
    def _check_omega(w, where, errors):
        try:
            w = float(w)
        except (TypeError, ValueError):
            return
        if not w > 0.0:
            errors.append(
                "%s: omega (%g) must be positive.  It is the products' Grueneisen "
                "coefficient and the only route the internal energy has into the "
                "pressure; at zero the equation of state stops reading e at all."
                % (where, w))

    def _resolve_materials(self, raw, errors):
        """
        Coerce each `Materials[]` entry through the same registry the cards use.

        An entry's sub-cards are declared in `params.py` exactly like a scene card is
        -- under the pseudo-card paths `Materials[]`, `Materials[].eos`,
        `Materials[].strength` and `Materials[].strength.plasticity` -- so this is the
        same walk `_read_one_card` does, and a named material cannot accept a type or
        a spelling the registry would reject anywhere else.

        The result is keyed by the FLAT canonical name of each parameter, which is the
        repository's one naming convention for a scene value; `materials.Material`
        then derives K, G, c_p, sigma_y0 and the EOS table row from it.  No formula
        derivation happens here.
        """
        out = {}
        mats = raw.get("Materials")
        if mats is not None and not isinstance(mats, list):
            return out
        for mat in mats or []:
            if not isinstance(mat, dict):
                continue
            name = mat.get("name")
            if not isinstance(name, str) or not name:
                continue
            where = "Materials[%r]" % name
            entry = {}
            self._coerce_into(mat, MATERIALS, where, entry, errors)

            eos = mat.get("eos")
            if isinstance(eos, dict):
                self._coerce_into(eos, MATERIALS_EOS, "%s.eos" % where, entry, errors)

            strength = mat.get("strength")
            if isinstance(strength, dict):
                self._coerce_into(strength, MATERIALS_STRENGTH,
                                  "%s.strength" % where, entry, errors)
                plasticity = strength.get("plasticity")
                if isinstance(plasticity, dict):
                    self._coerce_into(plasticity, MATERIALS_PLASTICITY,
                                      "%s.strength.plasticity" % where, entry, errors)

            damage = mat.get("damage")
            if isinstance(damage, dict):
                self._coerce_into(damage, MATERIALS_DAMAGE,
                                  "%s.damage" % where, entry, errors)
                if isinstance(damage.get("tension"), dict):
                    self._coerce_into(damage["tension"], MATERIALS_DAMAGE_TENSION,
                                      "%s.damage.tension" % where, entry, errors)
                entry["damage"] = damage
            out[name] = entry
        return out

    @staticmethod
    def _coerce_into(body, card, where, entry, errors):
        """Type-check one sub-card's fields into `entry`, keyed by flat canonical name.

        Unknown keys are not reported here: `_check_materials` has already named them,
        and reporting a key twice in the same error list is worse than not at all.
        """
        fields = card_fields(card)
        for field, value in body.items():
            param = fields.get(field)
            if param is None:
                continue
            coerced = _coerce(param, value, "%s.%s" % (where, field), errors)
            if coerced is not None:
                entry[param.name] = coerced

    def _build_materials(self, entries, errors, default_delta=0.0):
        """`{name: coerced entry}` -> `{name: materials.Material}`.

        Resolved in sorted-name order, so that the EOS table and therefore every
        particle's row index depend only on which materials a deck declares and not on
        the order the JSON happened to list them in.  An entry that cannot be resolved
        contributes its message to the same error list as everything else rather than
        raising out of the constructor, so a deck with two bad materials reports both.
        """
        out = {}
        for index, name in enumerate(sorted(entries)):
            try:
                out[name] = Material(name, index, entries[name], default_delta=default_delta)
            except MaterialError as exc:
                errors.append(str(exc))
        return out

    # ---------------------------------------------------------------- blocks --
    def _check_blocks(self, raw, errors, material_names=frozenset(), has_particles_file=False,
                      resolved=None):
        for section in ("FluidBlocks", "SolidBlocks"):
            blocks = raw.get(section)
            if blocks is not None and not isinstance(blocks, list):
                continue
            for i, block in enumerate(blocks or []):
                where = "%s[%d]" % (section, i)
                if not isinstance(block, dict):
                    errors.append("%s: expected an object." % where)
                    continue
                for key in block:
                    if key not in BLOCK_KEYS:
                        errors.append("%s: unknown key %r.%s"
                                      % (where, key, suggest(key, BLOCK_KEYS)))
                # When loading particles from file, start and end are not needed
                # (only objectId and material mapping are used)
                required_keys = ("objectId",) if has_particles_file else BLOCK_REQUIRED
                for key in required_keys:
                    if key not in block:
                        errors.append("%s: %s is required." % (where, key))
                if "shape" in block:
                    self._check_shape(block["shape"], "%s.shape" % where, errors,
                                      resolved or {})
                if "material" in block:
                    if block["material"] not in material_names:
                        errors.append(
                            "%s: unknown material %r; declared materials: %s.%s"
                            % (where, block["material"],
                               ", ".join(sorted(material_names)) or "(none)",
                               suggest(block["material"], material_names)))
                elif material_names:
                    # Every constant a particle needs comes from its material, and
                    # there is no default material to fall back on: a block that names
                    # none would have no density, no modulus and no equation of state.
                    errors.append(
                        "%s: material is required; every block names one of the "
                        "declared materials (%s).  There is no fallback: a material "
                        "constant exists only inside a Materials[] entry."
                        % (where, ", ".join(sorted(material_names))))

    @staticmethod
    def _check_shape(shape, where, errors, resolved=None):
        if not isinstance(shape, dict):
            errors.append("%s: expected an object." % where)
            return
        kind = str(shape.get("type", "")).lower()
        if kind not in SHAPE_KEYS:
            errors.append("%s: unknown shape type %r; known shapes: %s.%s"
                          % (where, shape.get("type"), ", ".join(sorted(SHAPE_KEYS)),
                             suggest(kind, SHAPE_KEYS)))
            return
        legal = SHAPE_KEYS[kind]
        for key in shape:
            if key not in legal:
                errors.append("%s: unknown key %r for shape %r.%s"
                              % (where, key, kind, suggest(key, legal)))
        for key in SHAPE_REQUIRED[kind]:
            if key not in shape:
                errors.append("%s: %s is required for shape %r." % (where, key, kind))
        if kind.startswith("dogbone"):
            # R >= D, or the arc never reaches the tab width (7).
            try:
                d = (float(shape["tabWidth"]) - float(shape["gaugeWidth"])) / 2.0
                if float(shape["radius"]) < d:
                    errors.append("%s: radius (%g) must be at least (tabWidth - gaugeWidth)/2 "
                                  "= %g, or the shoulder arc never reaches the tab width (7)."
                                  % (where, float(shape["radius"]), d))
            except (KeyError, TypeError, ValueError):
                pass
        # Which coordinate a dogbone's long axis is cannot be chosen by the scene: the
        # slab and the 3D bar are drawn along y, and the meridional half-section along x,
        # because an axisymmetric deck's axial direction is x and its y is a radius (11).
        # Pairing the wrong one with the geometry does not fail -- `dogbone` revolved about
        # its own width direction is a perfectly well-formed *tube* -- so it has to be
        # caught here or not at all.
        axisymmetric = bool((resolved or {}).get("axisymmetric"))
        if kind == "dogboneaxi" and not axisymmetric:
            errors.append("%s: shape 'dogboneaxi' is the meridional half-section of a "
                          "revolved bar and only means anything under "
                          "Domain.axisymmetric (11). Use 'dogbone' for a plane-strain "
                          "slab, or 'dogbone3d' for the bar in a full 3D lattice."
                          % where)
        if kind in ("dogbone", "dogbone3d") and axisymmetric:
            errors.append("%s: shape %r draws its long axis along y, but under "
                          "Domain.axisymmetric y is the RADIUS and x is the axial "
                          "coordinate (11), so revolving it gives a tube rather than a "
                          "bar -- silently, since the carve succeeds. Use 'dogboneaxi', "
                          "which is the same profile laid on its side."
                          % (where, kind))
        if kind in ("circle", "sphere", "cylinder"):
            try:
                if float(shape["radius"]) <= 0.0:
                    errors.append("%s: radius must be positive for shape %r, got %g."
                                  % (where, kind, float(shape["radius"])))
            except (KeyError, TypeError, ValueError):
                pass
        if kind == "cylinder":
            axis = str(shape.get("axis", "x")).lower()
            if axis not in ("x", "y", "z"):
                errors.append("%s: axis must be one of x, y, z for shape 'cylinder', "
                              "got %r." % (where, shape.get("axis")))

    def _check_constraints(self, raw, errors):
        constraints = raw.get("Constraints")
        if constraints is not None and not isinstance(constraints, list):
            return
        seen_names = {}
        for i, c in enumerate(constraints or []):
            where = "Constraints[%d]" % i
            if not isinstance(c, dict):
                errors.append("%s: expected an object." % where)
                continue
            for key in c:
                if key not in CONSTRAINT_KEYS:
                    errors.append("%s: unknown key %r.%s"
                                  % (where, key, suggest(key, CONSTRAINT_KEYS)))
            for key in CONSTRAINT_REQUIRED:
                if key not in c:
                    errors.append("%s: %s is required." % (where, key))
            name = c.get("name")
            if name is not None:
                if not isinstance(name, str) or not name.strip():
                    errors.append("%s: name must be a non-empty string, got %r." % (where, name))
                else:
                    clean_name = name.strip()
                    if clean_name in seen_names:
                        errors.append("%s: duplicate constraint name %r (first defined in Constraints[%d])."
                                      % (where, clean_name, seen_names[clean_name]))
                    else:
                        seen_names[clean_name] = i
            comp = c.get("components")
            if isinstance(comp, str) and (set(comp) - set("xyz") or not comp):
                errors.append("%s: components must be a non-empty subset of 'xyz', got %r."
                              % (where, comp))

    def _check_detonators(self, raw, errors):
        detonators = raw.get("Detonators")
        if detonators is not None and not isinstance(detonators, list):
            return
        for i, d in enumerate(detonators or []):
            where = "Detonators[%d]" % i
            if not isinstance(d, dict):
                errors.append("%s: expected an object." % where)
                continue
            for key in d:
                if key not in DETONATOR_KEYS:
                    errors.append("%s: unknown key %r.%s"
                                  % (where, key, suggest(key, DETONATOR_KEYS)))
            for key in DETONATOR_REQUIRED:
                if key not in d:
                    errors.append("%s: %s is required." % (where, key))
            pt = d.get("point")
            if pt is not None and not (isinstance(pt, (list, tuple)) and len(pt) == 3
                                       and all(isinstance(v, (int, float))
                                               and not isinstance(v, bool) for v in pt)):
                errors.append("%s: point must be a list of three numbers, got %r."
                              % (where, pt))
            t = d.get("time")
            if t is not None and (isinstance(t, bool)
                                  or not isinstance(t, (int, float))):
                errors.append("%s: time must be a number, got %r." % (where, t))

    def _check_reactive(self, raw, resolved, errors):
        """
        The cross-checks between the solver card and the declared materials.

        An `applies=` predicate is handed the resolved Configuration dict, which does
        not carry `Materials[]` -- so "is there an explosive in this scene at all", and
        "does any material declare a strength model", are not questions the registry
        can ask.  They are asked here instead, one layer later, beside the other checks
        that need the sections.

        A legacy deck that states its material constants flat has no `Materials[]` for
        any of this to read, so the JWL half also looks at the flat `EOS_TYPE`.
        """
        if raw.get("Materials") and not self.materials:
            # The section is there but did not resolve, and every question below is
            # about what the materials are.  Asking them anyway would report a missing
            # explosive that the deck plainly declares, on top of the real error.
            return

        legacy_jwl = str(resolved.get("EOS_TYPE") or "").lower() == "jwl"
        jwl_names = sorted(n for n, m in self.materials.items()
                           if m.eos_kind == EOS_JWL)
        any_jwl = bool(jwl_names) or legacy_jwl
        method = resolved.get("simulationMethod")

        if legacy_jwl and not self.materials:
            self._check_cj_density(resolved.get("JWL_RHO_CJ"),
                                   resolved.get("density0"), "Configuration", errors)
            self._check_omega(resolved.get("JWL_OMEGA"), "Configuration", errors)

        if any_jwl and method != "hypoElastic":
            errors.append(
                "a JWL material was declared, but Solver.type is %r.  The JWL branch "
                "lives in the solid solver: it needs the per-particle material table, "
                "the velocity gradient and Monaghan's pair viscosity, none of which the "
                "fluid branch has." % method)

        # The same argument, for the other p(rho, e) equation of state.  The fluid
        # solver's single global EOS is Tait, parameterised by c0, density0 and
        # exponent; a Mie-Gruneisen material has no exponent (derive_globals hands the
        # deck 1.0 as a fallback) and its whole point is the energy term Gamma0 rho e,
        # which the fluid branch carries no internal energy to evaluate.  Left
        # unchecked the deck would run -- as a LINEAR Tait fluid at the declared c0,
        # silently, with no error and no missing key -- which is the quiet-wrong-answer
        # failure section 18 keeps warning about rather than a crash.
        mg_names = sorted(n for n, m in self.materials.items()
                          if m.eos_kind == EOS_MIE_GRUNEISEN)
        if mg_names and method != "hypoElastic":
            errors.append(
                "material%s %s declare%s eos.type 'mie_gruneisen', but Solver.type is "
                "%r.  The Mie-Gruneisen branch lives in the solid solver: it needs the "
                "per-particle EOS table and the internal energy e, neither of which the "
                "fluid branch has, and the fluid branch's single Tait EOS would stand "
                "in for it silently.  Set Solver.type to 'hypoElastic', which runs a "
                "strengthless material perfectly well (G = 0)."
                % ("s" if len(mg_names) > 1 else "", ", ".join(mg_names),
                   "" if len(mg_names) > 1 else "s", method))

        # The fluid solver carries ONE equation of state, built from the global c0,
        # density0 and exponent, and it has no deviatoric stress at all.  Both of the
        # things a Materials[] deck can express beyond that are therefore errors under
        # it, and they are separate errors because the fixes are different: a solid
        # material means the deck wanted the solid solver, whereas two fluids means it
        # wanted something the fluid solver cannot do at all.
        if method == "deltaPlusSPH" and self.materials:
            solids = sorted(n for n, m in self.materials.items() if m.has_strength)
            if solids:
                errors.append(
                    "material%s %s declare%s a strength model, but Solver.type is "
                    "'deltaPlusSPH'.  A strength model means a shear modulus, a "
                    "deviatoric stress and a yield surface, none of which the fluid "
                    "solver has; set Solver.type to 'hypoElastic', which runs a "
                    "strengthless material perfectly well (G = 0)."
                    % ("s" if len(solids) > 1 else "", ", ".join(solids),
                       "" if len(solids) > 1 else "s"))
            elif len(self.materials) > 1:
                errors.append(
                    "%d materials were declared, but Solver.type is 'deltaPlusSPH', "
                    "which carries a single global equation of state (one c0, one "
                    "density0, one exponent) and no per-particle material table.  Only "
                    "the solid solver reads a material per particle."
                    % len(self.materials))

        n_det = len(raw.get("Detonators", []) or [])
        if any_jwl and n_det == 0:
            who = ", ".join(repr(n) for n in jwl_names) or "the flat Configuration"
            errors.append(
                "%s declares eos.type 'jwl', but the scene has no Detonators section. "
                " Under a programmed burn the lighting time is t_l = |x_0 - x_det|/D, "
                "so an explosive with no detonation point never lights and carries no "
                "pressure for the whole run." % who)
        if n_det and not any_jwl:
            errors.append(
                "Detonators were given, but no material declares eos.type 'jwl'.  A "
                "detonation point lights an explosive; there is nothing here for it "
                "to light.")

        # The Materials[] spelling of this is caught by MATERIAL_JWL_FORBIDDEN in
        # _check_materials, where a whole `strength` card is rejected; here it is the
        # flat legacy deck, where a yield point is just another Configuration key.
        if legacy_jwl and not self.materials and (
                resolved.get("yieldStress") is not None
                or resolved.get("yieldStrain") is not None):
            errors.append(
                "a yield point was given alongside eos.type 'jwl'.  A JWL material has "
                "no shear stiffness, so its deviatoric stress is identically zero and "
                "there is no trial deviator for a return map to correct.")

        if any_jwl and (resolved.get("GAMMA_SPH") or 0.0) > 0.0:
            errors.append(
                "Diffusion.gamma.coefficient is on alongside a JWL material.  The "
                "gamma-SPH correction is built from `gamma_pressure`, which is the "
                "GLOBAL Tait EOS -- so for an explosive particle it would difference "
                "two pressures off a curve that particle is not on.  Making it read "
                "the per-particle branch is the fix, and it has not been done.")

        if "Burn" in raw and not any_jwl:
            errors.append(
                "card Burn was given, but no material declares eos.type 'jwl'.  The "
                "burn fraction gates the JWL pressure branch and nothing else, so the "
                "card has no effect.")

    # ------------------------------------------------------------------- api --
    def get_cfg(self, name, enforce_exist=False):
        cfg = self.config["Configuration"]
        if enforce_exist:
            assert name in cfg, "%s is not set" % name
        return cfg.get(name)

    def get_fluid_blocks(self):
        return self.config.get("FluidBlocks", [])

    def get_solid_blocks(self):
        """Solid blocks are laid out exactly like fluid blocks; only the material tag
        and the constitutive treatment differ."""
        return self.config.get("SolidBlocks", [])

    def get_materials(self):
        """
        The declared materials, keyed by name, as resolved `materials.Material` objects.

        Every constant a particle needs is already derived on them -- rho0, K, G, the
        signal speed c_p, sigma_y0, the hardening modulus and the material's row of the
        EOS table -- so `SOLID.py` fills its per-particle fields by reading attributes
        rather than by re-deriving anything.  `Material.index` is the row index each
        particle carries.

        Empty for a legacy deck that states its material constants flat in a
        `Configuration` block instead; `SOLID.py` keeps its own path for that.
        """
        return self.materials

    def get_detonators(self):
        """
        Detonation points, read once at construction to set every explosive
        particle's lighting time t_l = min_d (t_d + |x_0 - x_d|/D).

            point   [x, y, z], world space, like a Constraints region
            time    when that point is lit; 0.0 if omitted

        Empty unless the scene declares the section, and a scene with a JWL
        material and no detonators is rejected rather than run dark.
        """
        return self.config.get("Detonators", [])

    def get_constraints(self):
        """
        Displacement boundary conditions, evaluated once against the initial positions.

            name         mandatory unique constraint identifier
            region       [[xlo, ylo, zlo], [xhi, yhi, zhi]]  selection box
            components   subset of "xyz": which components are prescribed
            displacement total displacement from x_0, reached at the end of the ramp
        """
        return self.config.get("Constraints", [])

    # ------------------------------------------------------------ provenance --
    def summary(self):
        """One line naming the scene and everything it set away from the default."""
        cfg = self.config["Configuration"]
        scene = [p.name for p in PARAMS if self.sources.get(p.name) == "scene"]
        return ("scene %s: %s material, %d parameters set, %d defaulted"
                % (os.path.basename(self.path), cfg.get("simulationMethod"),
                   len(scene), len(PARAMS) - len(scene)))

    def resolved_report(self):
        """The full resolved deck, one line per parameter, with where each came from.

        Worth writing next to the dumps of any run whose numbers are going to be
        quoted: CODE_DESCRIPTION has repeatedly had to annotate a measurement with
        "taken at PST_MODE sun, which the scenes no longer use".
        """
        lines = ["# resolved configuration for %s" % self.path]
        cfg = self.config["Configuration"]
        for card in CARD_ORDER:
            rows = [p for p in PARAMS if p.card[0] == card]
            if not rows:
                continue
            lines.append("")
            lines.append("[%s]" % card)
            for p in rows:
                lines.append("  %-28s = %-22r  (%s)"
                             % (".".join(p.card[1:] + (p.field,)), cfg.get(p.name),
                                self.sources.get(p.name, "default")))
        for flag in sorted(PRESENCE_FLAGS.values()):
            lines.append("  %-28s = %-22r  (%s)"
                         % (flag, cfg.get(flag), self.sources.get(flag, "default")))
        return "\n".join(lines)
