# Security Model

OpenAlphaLab / Calibration Engine is an offline Python library.

It does not:

- start a web server;
- open network ports;
- expose a public HTTP API;
- call remote APIs;
- require API keys, tokens or passwords;
- read `.env` files;
- execute shell commands from user inputs.

Inputs are local Python objects or local files such as JSON and CSV. Code runs only when the user imports the package and calls its functions from their own Python process.

## Formula Parser Safety

Alpha formulas are evaluated with an allowlisted AST parser, not Python `eval`.

Allowed expression features are intentionally narrow:

- numeric constants;
- local data names such as `close` and `volume`;
- allowlisted operators such as `rank`, `delta`, `ts_mean`;
- arithmetic operators `+`, `-`, `*`, `/`, `**`;
- unary `+` and `-`.

The parser rejects:

- imports;
- builtins;
- attribute access;
- subscript access;
- keyword arguments;
- arbitrary function calls.

## Reporting Issues

If you find a security issue, do not publish exploit details in a public issue. Report the minimal reproduction privately to the repository owner.
