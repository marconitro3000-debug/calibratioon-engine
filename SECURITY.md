# Security Model

Calibration Engine is an offline Python library.

It does not:

- start a web server;
- open network ports;
- expose a public HTTP API;
- call remote APIs;
- require API keys, tokens or passwords;
- read `.env` files;
- execute shell commands from user inputs.

Inputs are local Python objects passed by your own code. The library runs only when you import it and call it from your own Python process.

The project intentionally avoids networking, credential handling and command execution.
