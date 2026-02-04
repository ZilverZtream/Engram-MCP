import pytest

from engram_mcp.security import validate_project_field


def test_validate_project_field_accepts_printable():
    assert validate_project_field("Project-1", field_name="project_name") == "Project-1"


def test_validate_project_field_rejects_control_chars():
    with pytest.raises(ValueError):
        validate_project_field("bad\nname", field_name="project_name")
