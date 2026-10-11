# macOS USB authorization

Start the desktop or web viewer as your normal user from a terminal:

```sh
.venv/bin/topdon-duo-desktop --rotate 90
# Or:
.venv/bin/topdon-duo --ambient 21.9 --rotate 90
```

The real Duo backend on macOS starts a separate USB helper through `/usr/bin/sudo`.
Enter your password in the launching terminal if requested. The helper runs the
configured interpreter in isolated mode (`-I`) with bytecode writes disabled (`-B`),
with a minimal environment and `/`
as its working directory. It imports only the USB capture path and its dependencies;
it does not initialize the UI, Custom packages, ONNX sessions or model downloads.
Linux retains its existing direct capture and udev rule. Injected/future backends
are unaffected. An explicitly elevated legacy launch retains direct Duo capture.

The helper owns configuration, driver detachment, negotiation, queued bulk reads
and bounded Duo control-interface transfers. Complete raw frames cross private
stdin/stdout pipes in a length-bounded binary protocol with JSON metadata. There
is no pickle, shell-command dispatch, path selection or application-code execution
request. The viewer continuously drains the helper and retains the newest frame.
The existing HardwareControls owner stays in the viewer and performs the same
startup reads, write/readback checks and settings restoration before shutdown.
Closing the command pipe stops acquisition, drains transfers and closes the camera,
including driver reattachment. An abrupt viewer failure cannot guarantee hardware
settings restoration; reconnect the camera if it remains unavailable.

Authorization requires a controlling terminal. For an IDE launch that cannot show
the password prompt, use a terminal instead; `sudo -v` can authorize that terminal
before launching. Failed authorization produces a viewer error. This implementation
does not install a background daemon, modify sudoers or grant passwordless access.

## Development and distribution

This is a development helper authorized for each launch using your trusted project
checkout and Python environment. Those files and dependencies are user-writable.
Do not install this interpreter/checkout as a permanent privileged service or add
it to a passwordless sudo rule. A production helper needs an administrator-owned,
fixed installation and authenticated client access, with signed app packaging and
an approved service installation mechanism such as SMAppService.

Routine viewer execution is unprivileged; USB takeover still requires authorization.
Removing privileged USB execution altogether needs a separately validated macOS
driver integration. Apple's restricted VM capture entitlement is not a general
solution for this viewer.

## Existing root-owned model downloads

Models downloaded by earlier `sudo` viewer sessions may be owned by root with
mode `0600`. New user-process downloads do not create that ownership mismatch.
To retain an affected existing model, inspect its ownership and repair only that
file. For example, the affected RealESRGAN file in this project's review was:

```sh
ls -l "$HOME/Library/Caches/topdon-duo/models/realesr-general-x4v3-a946f7a93970.onnx"
sudo chown "$(id -u):$(id -g)" "$HOME/Library/Caches/topdon-duo/models/realesr-general-x4v3-a946f7a93970.onnx"
```

Do not recursively change unrelated caches or application data. Ownership repair
does not bypass the downloader's existing checksum verification.

## Verification

`tests/test_macos_usb.py` exercises the actual client/service protocol over local
pipes with fake USB: frame bytes, controls, USB errors, malformed requests,
disconnect cleanup and platform/factory selection. It runs without sudo or hardware.
A physical macOS check must additionally verify authorization, sustained capture,
control readback/restoration and driver release with other camera apps closed.
