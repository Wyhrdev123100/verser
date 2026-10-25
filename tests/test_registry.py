from uatp.models import REGISTRY


def test_adapters_registered():
    assert "toy_gpt" in REGISTRY.list_adapters()
    assert "huggingface" in REGISTRY.list_adapters()


def test_models_registered():
    names = {m.name for m in REGISTRY.list_models()}
    assert "toy-gpt" in names


def test_get_adapter():
    a = REGISTRY.get_adapter("toy_gpt")
    assert a.name == "toy_gpt"


def test_unknown_raises():
    try:
        REGISTRY.get_adapter("nope")
        assert False
    except KeyError:
        assert True