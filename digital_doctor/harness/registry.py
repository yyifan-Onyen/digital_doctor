"""Version-aware registry for executable clinical skill bundles."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Dict, Iterable, List, Optional

from ..skills.base import ClinicalSkill


class SkillNotFoundError(LookupError):
    pass


class SkillRegistry:
    def __init__(self, skills: Iterable[ClinicalSkill] = ()) -> None:
        self._skills: Dict[str, Dict[str, ClinicalSkill]] = defaultdict(dict)
        for skill in skills:
            self.register(skill)

    def register(self, skill: ClinicalSkill) -> None:
        manifest = skill.manifest
        versions = self._skills[manifest.skill_id]
        if manifest.version in versions:
            raise ValueError(
                f"Skill {manifest.skill_id!r} version {manifest.version!r} is already registered"
            )
        versions[manifest.version] = skill

    def resolve(self, skill_id: str, version: Optional[str] = None) -> ClinicalSkill:
        versions = self._skills.get(skill_id, {})
        if not versions:
            raise SkillNotFoundError(f"Unknown clinical skill: {skill_id}")
        if version is not None:
            try:
                return versions[version]
            except KeyError as exc:
                raise SkillNotFoundError(
                    f"Unknown version {version!r} for clinical skill {skill_id!r}"
                ) from exc
        def version_key(value: str):
            return tuple(
                (0, int(part)) if part.isdigit() else (1, part)
                for part in re.split(r"[._+-]", value)
            )

        latest = max(versions, key=version_key)
        return versions[latest]

    def manifests(self) -> List[Dict[str, object]]:
        return [
            skill.manifest.to_dict()
            for skill_id in sorted(self._skills)
            for _, skill in sorted(self._skills[skill_id].items())
        ]


__all__ = ["SkillNotFoundError", "SkillRegistry"]
