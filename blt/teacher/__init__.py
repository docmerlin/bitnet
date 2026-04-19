"""Teacher model adapters for BLT distillation."""

from blt.teacher.facebook_blt import FacebookBLTTeacher, import_upstream_blt

__all__ = ["FacebookBLTTeacher", "import_upstream_blt"]
