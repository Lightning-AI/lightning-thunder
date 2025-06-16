import thunder.examine
import torch


def test_examine_fn():
    def foo(x):
        x[0] = 5 * x[1]

    x = torch.ones(2, 2)
    thunder.examine.examine(foo, x)


def test_examine_jfn():
    def foo(x):
        x[0] = 5 * x[1]

    jfoo = thunder.jit(foo)
    x = torch.ones(2, 2)
    thunder.examine.examine(jfoo, x)


def test_examine_noncallable(capsys):
    x = torch.ones(2, 2)
    y = torch.ones(2, 2)
    thunder.examine.examine(x, y)
    captured = capsys.readouterr()
    assert "expected `fn` to be a callable" in captured.out
