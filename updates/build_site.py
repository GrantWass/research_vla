"""Wrap notebook.html (an artifact fragment) into a standalone page at site/index.html for Vercel."""
import pathlib

here = pathlib.Path(__file__).parent
src = (here / "notebook.html").read_text()
head, sep, body = src.partition("</style>")
assert sep, "notebook.html has no </style>"
page = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    + head + "</style>\n"
    "<style>:root{color-scheme:light}body{margin:0}[hidden]{display:none!important}</style>\n"
    "</head>\n<body>" + body + "\n</body>\n</html>\n"
)
out = here / "site"
out.mkdir(exist_ok=True)
(out / "index.html").write_text(page)
print("wrote", out / "index.html")
