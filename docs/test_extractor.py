#!/usr/bin/env python

import json
import shutil
import tempfile
from pathlib import Path

from extractor import SupportMatrixExtractor, managed_container

import docker

IMG = "python:3.8-buster@sha256:ccc66c06817c2e5b7ecd40db1c4305dea3cd9e48ec29151a593e0dbd76af365e"

DATAJSON = Path(__file__).parent / "fixtures" / "data.json"
SAMPLEJSON = Path(__file__).parent / "fixtures" / "sample.json"
SAMPLEAFTERJSON = Path(__file__).parent / "fixtures" / "sample_after.json"


def test_get_from_envfile():
    ETC_OS_RELEASE = "/etc/os-release"
    with tempfile.TemporaryFile() as f, managed_container(IMG) as c:
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.use_container(c)
        xtr.get_from_envfile(ETC_OS_RELEASE, "PRETTY_NAME")
        assert (
            xtr.data["x"]["22.02"].get("PRETTY_NAME") == "Debian GNU/Linux 10 (buster)"
        )

        xtr.get_from_envfile(ETC_OS_RELEASE, "PRETTY_NAME", "os")
        assert xtr.data["x"]["22.02"].get("os") == "Debian GNU/Linux 10 (buster)"

        xtr.get_from_envfile(ETC_OS_RELEASE, "BAZ")
        assert xtr.data["x"]["22.02"].get("BAZ") == SupportMatrixExtractor.ERROR


def test_get_from_pip():
    with tempfile.TemporaryFile() as f, managed_container(IMG) as c:
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.use_container(c)
        xtr.get_from_pip("pip")
        assert xtr.data["x"]["22.02"].get("pip") == "22.0.4"

        xtr.get_from_pip("spam")
        assert xtr.data["x"]["22.02"].get("spam") == SupportMatrixExtractor.ERROR


def test_get_from_env():
    with tempfile.TemporaryFile() as f, managed_container(IMG) as c:
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.use_container(c)
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


def test_get_from_image():
    with tempfile.TemporaryFile() as f, managed_container(IMG) as c:
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.use_container(c)
        xtr.get_from_image("Size", "size")
        assert xtr.data["x"]["22.02"].get("size") != SupportMatrixExtractor.ERROR

        xtr.get_from_image("blah")
        assert xtr.data["x"]["22.02"].get("blah") == SupportMatrixExtractor.ERROR


def test_get_from_cmd():
    with tempfile.TemporaryFile() as f, managed_container(IMG) as c:
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.use_container(c)
        xtr.get_from_cmd("cat /proc/1/cmdline", "cmdline")
        assert xtr.data["x"]["22.02"].get("cmdline") != SupportMatrixExtractor.ERROR

        xtr.get_from_cmd("cat /proc/0/x", "cmderr")
        assert xtr.data["x"]["22.02"].get("cmderr") == SupportMatrixExtractor.ERROR


def test_insert_snippet():
    with tempfile.TemporaryFile() as f, managed_container(IMG) as c:
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.use_container(c)
        xtr.insert_snippet("release", "99.99")
        assert "99.99" in xtr.data["x"]["22.02"].get("release")


def test_to_json():
    with tempfile.NamedTemporaryFile() as f, managed_container(IMG) as c:
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.use_container(c)
        xtr.get_from_pip("pip")
        assert xtr.to_json() == r'{"x": {"22.02": {"pip": "22.0.4"}}}'


def test_from_json():
    with tempfile.NamedTemporaryFile() as f, managed_container(IMG) as c:
        shutil.copyfile(DATAJSON, f.name)
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.use_container(c)
        xtr.from_json()
        print(f"{xtr.data}")
        assert "first" in xtr.data.keys()
        assert "one" in xtr.data["first"]

        xtr.get_from_env("SHELL", "first")
        assert "first" in xtr.data.keys()
        assert "one" in xtr.data["first"].keys()
        assert "x" in xtr.data.keys()
        assert "22.02" in xtr.data["x"]


def test_to_json_file():
    with tempfile.NamedTemporaryFile() as f, managed_container(IMG) as c:
        shutil.copyfile(SAMPLEJSON, f.name)
        xtr = SupportMatrixExtractor("x", "22.02", f.name)
        xtr.use_container(c)
        xtr.get_from_pip("pip")
        xtr.from_json()
        xtr.to_json_file()
        a, b = "", ""
        with open(SAMPLEAFTERJSON) as fa, open(f.name) as fb:
            a = json.load(fa)
            b = json.load(fb)
            assert a == b


def test_managed_container():
    with managed_container("foo-bar:baz") as nf:
        assert isinstance(nf, docker.errors.ImageNotFound)
        assert nf.status_code == 404
