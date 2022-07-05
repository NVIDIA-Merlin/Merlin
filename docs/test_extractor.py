#!/usr/bin/env python

import json
import os
import shutil
import tempfile
from pathlib import Path

from extractor import SupportMatrixExtractor

DATAJSON = Path(__file__).parent / "fixtures" / "data.json"
SAMPLEJSON = Path(__file__).parent / "fixtures" / "sample.json"
SAMPLEAFTERJSON = Path(__file__).parent / "fixtures" / "sample_after.json"


def test_get_from_envfile():
    ETC_OS_RELEASE = "/etc/os-release"
    with tempfile.TemporaryFile() as f:
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.get_from_envfile(ETC_OS_RELEASE, "PRETTY_NAME")
        assert (
            xtr.data["x"]["22.02"].get("PRETTY_NAME") == "Debian GNU/Linux 10 (buster)"
        )

        xtr.get_from_envfile(ETC_OS_RELEASE, "PRETTY_NAME", "os")
        assert xtr.data["x"]["22.02"].get("os") == "Debian GNU/Linux 10 (buster)"

        xtr.get_from_envfile(ETC_OS_RELEASE, "BAZ")
        assert xtr.data["x"]["22.02"].get("BAZ") == SupportMatrixExtractor.ERROR


def test_get_from_pip():
    with tempfile.TemporaryFile() as f:
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.get_from_pip("pip")
        assert xtr.data["x"]["22.02"].get("pip") == "22.0.4"

        xtr.get_from_pip("spam")
        assert xtr.data["x"]["22.02"].get("spam") == SupportMatrixExtractor.ERROR


def test_get_from_python():
    with tempfile.TemporaryFile() as f:
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.get_from_python("json")
        assert xtr.data["x"]["22.02"].get("json") == "2.0.9"

        xtr.get_from_python("spam")
        assert xtr.data["x"]["22.02"].get("spam") == SupportMatrixExtractor.ERROR


def test_get_from_env():
    with tempfile.TemporaryFile() as f:
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.get_from_env("SHELL")
        assert xtr.data["x"]["22.02"].get("SHELL") != SupportMatrixExtractor.ERROR

        xtr.get_from_env("SHELL", "shell")
        assert "shell" in xtr.data["x"]["22.02"].keys()
        assert xtr.data["x"]["22.02"].get("shell") != SupportMatrixExtractor.ERROR

        xtr.get_from_env("BAZ")
        assert "BAZ" in xtr.data["x"]["22.02"].keys()
        assert xtr.data["x"]["22.02"].get("BAZ") == SupportMatrixExtractor.ERROR

        xtr.get_from_env("BAZ", "bar")
        assert "bar" in xtr.data["x"]["22.02"].keys()
        assert xtr.data["x"]["22.02"].get("bar") == SupportMatrixExtractor.ERROR

        os.environ["SMX_COMPRESSED_SIZE"] = "7169310137"
        xtr.get_from_env("SMX_COMPRESSED_SIZE", "compressedSize")
        assert "compressedSize" in xtr.data["x"]["22.02"].keys()
        assert xtr.data["x"]["22.02"].get("compressedSize") == "6.68 GB"
        del os.environ["SMX_COMPRESSED_SIZE"]


def test_get_from_cmd():
    with tempfile.TemporaryFile() as f:
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.get_from_cmd("cat /proc/1/cmdline", "cmdline")
        assert xtr.data["x"]["22.02"].get("cmdline") != SupportMatrixExtractor.ERROR

        xtr.get_from_cmd("cat /proc/0/x", "cmderr")
        assert xtr.data["x"]["22.02"].get("cmderr") == SupportMatrixExtractor.ERROR


def test_insert_snippet():
    with tempfile.TemporaryFile() as f:
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.insert_snippet("release", "99.99")
        assert "99.99" in xtr.data["x"]["22.02"].get("release")


def test_to_json():
    with tempfile.NamedTemporaryFile() as f:
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.get_from_pip("pip")
        assert xtr.to_json() == r'{"x": {"22.02": {"pip": "22.0.4"}}}'


def test_from_json():
    with tempfile.NamedTemporaryFile() as f:
        shutil.copyfile(DATAJSON, f.name)
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.from_json()
        print(f"{xtr.data}")
        assert "first" in xtr.data.keys()
        assert "one" in xtr.data["first"]

        xtr.get_from_env("SHELL", "first")
        assert "first" in xtr.data.keys()
        assert "one" in xtr.data["first"].keys()
        assert "x" in xtr.data.keys()
        assert "22.02" in xtr.data["x"]

    with tempfile.NamedTemporaryFile() as f:
        shutil.copyfile(DATAJSON, f.name)
        xtr = SupportMatrixExtractor("first", "one", f.name)
        xtr.from_json()
        print(f"{xtr.data}")
        assert "first" in xtr.data.keys()
        assert "one" in xtr.data["first"]
        assert xtr.data["first"]["one"] == "1"
        assert "two" in xtr.data["first"]
        assert xtr.data["first"]["two"] == "2"


def test_to_json_file():
    with tempfile.NamedTemporaryFile() as f:
        shutil.copyfile(SAMPLEJSON, f.name)
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.from_json()
        xtr.get_from_pip("pip")
        xtr.to_json_file()
        a, b = "", ""
        with open(SAMPLEAFTERJSON, encoding="utf-8") as fa, open(
            f.name, encoding="utf-8"
        ) as fb:
            a = json.load(fa)
            b = json.load(fb)
            assert a == b

    with tempfile.NamedTemporaryFile() as f:
        shutil.copyfile(SAMPLEJSON, f.name)
        xtr = SupportMatrixExtractor("x", "22.02", f.name, force=True)
        xtr.from_json()
        xtr.get_from_pip("pip")
        xtr.to_json_file()
        a, b = "", ""
        with open(SAMPLEAFTERJSON, encoding="utf-8") as fa, open(
            f.name, encoding="utf-8"
        ) as fb:
            a = json.load(fa)
            b = json.load(fb)
            assert a == b


def test_already_present():
    # First test intentionally does not copy a data.json file.
    xtr = SupportMatrixExtractor("merlin-training", "22.02", "blah")
    assert xtr.already_present() is False

    with tempfile.NamedTemporaryFile() as f:
        shutil.copyfile(SAMPLEJSON, f.name)
        xtr = SupportMatrixExtractor("merlin-training", "22.02", f.name)
        xtr.from_json()
        assert xtr.already_present() is True

    with tempfile.NamedTemporaryFile() as f:
        shutil.copyfile(SAMPLEJSON, f.name)
        xtr = SupportMatrixExtractor("merlin-training", "22.02", f.name, force=True)
        xtr.from_json()
        assert xtr.already_present() is False

    with tempfile.NamedTemporaryFile() as f:
        shutil.copyfile(SAMPLEJSON, f.name)
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.from_json()
        assert xtr.already_present() is False

    with tempfile.NamedTemporaryFile() as f:
        shutil.copyfile(DATAJSON, f.name)
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.from_json()
        assert "first" in xtr.data.keys()
        assert "one" in xtr.data["first"]

        xtr.get_from_env("SHELL", "first")
        assert "first" in xtr.data.keys()
        assert "one" in xtr.data["first"].keys()
        assert "x" in xtr.data.keys()
        assert "22.02" in xtr.data["x"]
