from aistudio_api.infrastructure.gateway.session import (
    AI_STUDIO_LOW_POWER_UI_JS,
    _install_low_power_ui_sync,
    _low_power_ui_active_sync,
)


class FakeContext:
    def __init__(self) -> None:
        self.scripts: list[str] = []

    def add_init_script(self, script: str) -> None:
        self.scripts.append(script)


class FakePage:
    def __init__(self, result=True, raises=False) -> None:
        self.result = result
        self.raises = raises
        self.expressions: list[str] = []

    def evaluate(self, expression: str):
        self.expressions.append(expression)
        if self.raises:
            raise RuntimeError("page closed")
        return self.result


def test_low_power_ui_is_installed_as_context_init_script() -> None:
    context = FakeContext()

    _install_low_power_ui_sync(context)

    assert context.scripts == [AI_STUDIO_LOW_POWER_UI_JS]


def test_low_power_ui_only_targets_upgrade_card_animation() -> None:
    assert ".upgrade-card-wrapper" in AI_STUDIO_LOW_POWER_UI_JS
    assert "display: none !important" in AI_STUDIO_LOW_POWER_UI_JS
    assert "document.getAnimations" not in AI_STUDIO_LOW_POWER_UI_JS
    assert "requestAnimationFrame" not in AI_STUDIO_LOW_POWER_UI_JS


def test_low_power_ui_active_checks_style_marker() -> None:
    page = FakePage(result=True)

    assert _low_power_ui_active_sync(page) is True
    assert page.expressions == [
        "!!document.getElementById('aistudio-api-low-power-ui')"
    ]


def test_low_power_ui_active_handles_closed_page() -> None:
    assert _low_power_ui_active_sync(FakePage(raises=True)) is False
