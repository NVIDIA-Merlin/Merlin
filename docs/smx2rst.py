#!/usr/bin/env python3

# Read the data.json file.
# Create an RST file for each container.
# (Each RST file can be literalincluded into a file with headings and body text.)
#
# ==========  =============  =============
#             22.03          22.02
# ==========  =============  =============
# DGX
# ----------------------------------------
# DGX System  * DGX-1        * DGX-1
#             * DGX-2        * DGX-2
#             * DGX A100     * DGX A100
#             * DGX Station  * DGX Station
# ==========  =============  =============
#
# Or, a list table
#
#

import argparse
import json
import logging
import os
import re
import sys
import typing
from pathlib import Path

import yaml
from mergedeep import merge

level = logging.DEBUG if os.environ.get("DEBUG") else logging.INFO
logging.basicConfig(level=level)
logger = logging.getLogger("smx2rst")


class Smx2Rst:

    data_json_path: str
    data = {}
    table_config = {}
    release_pattern = re.compile(r"(\d{2})\.(\d{2})")  # YY.MM

    def __init__(self, json_file: str):
        self.data_json_path = os.path.abspath(json_file)
        if not os.path.exists(self.data_json_path) or not os.path.isfile(
            self.data_json_path
        ):
            logger.info("File is not found or is not a file: %s", self.data_json_path)
            sys.exit(1)

    def read_table_config(self, path: str):
        yaml_conf = os.path.abspath(path)
        logger.info("Opening YAML config file: %s", yaml_conf)
        with open(yaml_conf) as f:
            documents = yaml.safe_load_all(f)
            for conf in documents:
                self.table_config = merge({}, conf, self.table_config)

    def from_json(self):
        """Read the data.json file that was created with extractor.py.

        The JSON file has a key for each container name and subordinate
        keys for each release. Within each release dictionary, the JSON
        file has a key and value pair for the information to show
        in the support matrix table.
        """
        logger.info("Opening JSON file: %s", self.data_json_path)
        with open(self.data_json_path) as f:
            self.data = json.load(f)

    def to_rst(self, path: str):
        """Write the RST files that have the support matrix tables for
        each container.

        The implementation is to iterate over the containers from
        the JSON file and create one file for each container.

        Parameters
        ----------
        path : str
            The output path for the RST table snippets.
            The directory is created if it does not exist.
        """

        outdir = Path(path)
        if not outdir.exists():
            logger.info("Creating output directory: %s", str(outdir))
            outdir.mkdir(parents=True, exist_ok=True)
            logger.info("   ...done.")

        for container in self.data.keys():
            years = [
                self.release_pattern.search(x).group(1)
                for x in self.data[container].keys()
            ]
            years = sorted(set(years), reverse=True)

            fname = container.replace("/", "-")
            outpath = outdir / (fname + ".rst")
            logger.info("Creating RST table snippet: %s", str(outpath))
            with open(outpath, "w") as f:
                for year in years:
                    table = self.table_as_str(container, year)
                    for line in table:
                        f.write(line)
                logger.info("   ...done.")

    # Though I agree with pylint that this needs a refactor...not today.
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def table_as_str(self, container: str, rel_prefix: str) -> typing.List[str]:
        """Return the support matrix table for the specified container
        in RST simple table form.
        """

        # This is a two-pass implementation. The first pass iterates
        # over the data and collects the maximum dimensions for the table:
        # - number of columns, keeping in mind there is a stub column with row "headings"
        # - for each column, the maximum string length, all rows are padded with spaces to
        #   this length
        # - within each row, the maximum number of lines--specifically for bulleted lists
        #
        self.set_table_dims(container, rel_prefix)
        to_ret = []

        # Make the stub row headings.
        fields = self.table_config[container]
        colwidth = fields["maxwidth"]
        to_ret.append("=" * colwidth + "  ")
        to_ret.append("Container Release".ljust(colwidth) + "  ")
        to_ret.append("=" * colwidth + "  ")

        for k, v in fields.items():
            if k == "maxwidth":
                continue
            # Print the stub row heading.
            bold_stub = "**{}**".format(k)
            to_ret.append(bold_stub.ljust(colwidth) + "  ")

            # If this row has multiple lines, add the blank-padded lines.
            stub_line_count = k.count("\n")
            for _ in range(v["maxlines"] - stub_line_count):
                to_ret.append(" " * colwidth + "  ")

            if "span" in v.keys() and v["span"] is True:
                to_ret.append("-" * colwidth + "--")

        to_ret.append("=" * colwidth + "  ")
        # End stub row headings.

        # Print a column for each release.
        year_data = {
            k: v for k, v in self.data[container].items() if k.startswith(rel_prefix)
        }
        for idx, release in enumerate(sorted(year_data, reverse=True)):
            data = year_data.get(release)
            colwidth = data["maxwidth"]
            relhead = "Release " + release

            # Print the release, this is not part of the fields dictionary.
            to_ret[0] += "=" * colwidth + "  "
            to_ret[1] += relhead.ljust(colwidth) + "  "
            to_ret[2] += "=" * colwidth + "  "

            line_no = 3
            for k, v in fields.items():
                if k == "maxwidth":
                    continue

                if "field" in v.keys():
                    val = data[v["field"]]
                    for line in val.split("\n"):
                        to_ret[line_no] += line.ljust(colwidth)
                        if len(year_data) - 1 == idx:
                            to_ret[line_no] += "\n"
                        else:
                            to_ret[line_no] += "  "
                        line_no += 1

                    for _ in range(v["maxlines"] - val.count("\n")):
                        to_ret[line_no] += " " * colwidth
                        if len(year_data) - 1 == idx:
                            to_ret[line_no] += "\n"
                        else:
                            to_ret[line_no] += "  "
                        line_no += 1

                if "span" in v.keys():
                    to_ret[line_no] += " " * colwidth
                    line_no += 1
                    to_ret[line_no] += "-" * colwidth
                    line_no += 1

                    # If this isn't the last column, add two spaces
                    # or dashes.
                    if len(year_data) - 1 != idx:
                        to_ret[line_no - 2] += "  "
                        to_ret[line_no - 1] += "--"

            to_ret[line_no] += "=" * colwidth + "  "

        for idx, line in enumerate(to_ret):
            to_ret[idx] = "   " + line + "\n"

        table_heading = rel_prefix + ".xx Container Images"

        # In reverse order
        to_ret.insert(0, "\n")
        to_ret.insert(0, "   :align: left\n")
        to_ret.insert(0, ".. table::\n")
        to_ret.insert(0, "\n")
        to_ret.insert(0, "~" * len(table_heading) + "\n")
        to_ret.insert(0, table_heading + "\n")
        to_ret.insert(0, "\n")

        # Final blank line
        to_ret.append("\n")

        return to_ret

    # Right now, the foll code can indicate the number of rows. (But that's not
    # super helpful because I do not know the max number of lines in the multi-line
    # fields like NVIDIA Driver.)
    # The lone useful information is the length of the longes stub heading.
    # This isn't general, but is a general problem.  I need to know the maximum
    # number of lines in each row--across releases of the container--and the longest string
    # in each column.
    #
    # The stub column is 26 chars wide for "Container Operating System"
    # I didn't calculate it, but each row is one line.
    # For starters, get the max column lengths.  The key to max lines in a row is count("\n").
    #
    # For each release (shown in single column), we need the widest string
    # and a list of newline counts.
    def set_table_dims(self, container: str, rel_prefix: str):
        # The first column shows the stub headings, "Release" is not included in `fields`.
        # The release value, like 22.03, is always a single line.
        fields = self.table_config[container]
        stubs = list(fields.keys())
        stub_maxwidth = len(max(["Release"] + stubs, key=len))

        # Seems unlikely that the row stub "heading" has the most lines, but possible.
        for field in fields.keys():
            if field == "maxwidth":
                continue
            fields[field]["maxlines"] = field.count("\n")
        # components = { k: v for k, v in fields.items() if not "span" in v.keys() }
        # Filter decorative span rows.

        releases = [x for x in self.data[container] if x.startswith(rel_prefix)]
        # releases = list(self.data[container])
        for release in releases:
            # Find the longest string that we'll put in a column for this release.
            maxwidth = 0
            for k, v in fields.items():
                if k == "maxwidth" or "span" in v.keys():
                    continue
                val = self.data[container][release][v["field"]]
                line_len = max(len(x) for x in val.split("\n"))
                line_count = val.count("\n")
                maxwidth = max(maxwidth, line_len)
                v["maxlines"] = max(v["maxlines"], line_count)

            self.data[container][release]["maxwidth"] = maxwidth
        fields["maxwidth"] = stub_maxwidth + len("**") + len("**")


def main(args):
    file = args.file
    config = args.config
    outdir = args.dir

    if not file:
        file = Path(__file__).parent / "data.json"
    if not config:
        config = Path(__file__).parent / "table_config.yaml"
    if not outdir:
        outdir = Path(__file__).parent / "source" / "generated"

    smx2rst = Smx2Rst(file)
    smx2rst.read_table_config(config)
    smx2rst.from_json()
    smx2rst.to_rst(outdir)


def parse_args():
    """
    python smx2rst.py -f data.json
    """
    parser = argparse.ArgumentParser(description=("Merlin Support Matrix Tool"))

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="JSON data file",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="YAML configuration file",
    )

    parser.add_argument(
        "-d", "--dir", type=str, help="Output directory for RST table snippets"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
