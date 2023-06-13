import argparse, os, shutil
from parseWalker import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "--template-directory",
    nargs="?",
    type=str,
    help="Path for the template files",
    default="Xyz-template",
    dest="template_dir",
)
parser.add_argument(
    "--destination-directory",
    nargs="?",
    type=str,
    help="Path for the generated files",
    default="Xyz-gen",
    dest="dest_dir",
)
parser.add_argument(
    "--declarative-input-file",
    nargs="?",
    type=str,
    help="Path for the input declarative file",
    default="declr",
    dest="input_file",
)
parser.add_argument(
    "--output-caller-file",
    nargs="?",
    type=str,
    help="Path for the output caller file; if not given, output to stdout",
    default="",
    dest="output_file",
)
parser.add_argument(
    "--includes",
    nargs="+",
    type=str,
    help="Extra Include files which should be contained in iterator.h"
    " as custom algorithm, must already be presented in template directory",
    default=[],
    dest="includes",
)
args = parser.parse_args()

if not os.path.exists(args.template_dir):
    raise ValueError("Template Directory doesn't exist")
if os.path.exists(args.dest_dir):
    shutil.rmtree(args.dest_dir)
if not os.path.exists(args.input_file):
    raise ValueError("Input File doesn't exist")

shutil.copytree(args.template_dir, args.dest_dir)

includes = []
for inc in args.includes:
    if not os.path.exists(os.path.join(args.template_dir, inc)):
        raise ValueError("Include File doesn't exist")
    includes.append(f'#include "{inc}"\n')

with open(args.input_file, "r", encoding="utf-8") as in_declr:
    lib = GrowingLibrary(
        in_declr.read(), args.template_dir, args.dest_dir, args.output_file, includes
    )
    print(lib)
