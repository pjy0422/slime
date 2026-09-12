from pathlib import Path


ROOT = Path(__file__).parents[1]
PATCH = ROOT / "dtap_integration/patches/p5-windows-macos-placement.patch"


def test_platform_patch_is_last_managed_overlay():
    apply_script = (ROOT / "dtap_integration/apply.sh").read_text(encoding="utf-8")
    platform = '"$script_dir/patches/p5-windows-macos-placement.patch"'
    assert platform in apply_script
    assert apply_script.index(platform) > apply_script.index('"$script_dir/patches/p4-h2-live-stability.patch"')


def test_platform_patch_keeps_guest_verification_and_startup_fail_closed():
    patch = PATCH.read_text(encoding="utf-8")
    assert '"windows-injection": frozenset({' in patch
    assert '"macos-injection": frozenset({' in patch
    assert '"inject_registry"' in patch
    assert '"inject_cron_job"' in patch
    assert "Get-ItemPropertyValue" in patch
    assert "crontab -l" in patch
    assert "Skylake-Client-v4" in patch
    assert "prepared baseline" in patch
    assert "external overlay cannot expose an internal VM-state snapshot" in patch
    assert "_inject_text_file(file_path, full)" in patch
    assert "MCP-decorated FunctionTool objects are not callable" in patch
    assert "healthy = await self._wait_for_healthy" in patch
    assert "if not healthy:" in patch


def test_platform_patch_uses_only_action_scoped_readback_targets():
    patch = PATCH.read_text(encoding="utf-8")
    assert 'path = str(kwargs["file_path"])' in patch
    assert 'path = str(kwargs["registry_path"])' in patch
    assert 'path = str(kwargs["target_vm_path"])' in patch
    assert "shlex.quote(path)" in patch
    assert "Get-ChildItem" not in patch
    assert '"find /' not in patch
