# Installation and first launch

[Documentation home](../README.md) · [Capture](capture-guide.md) · [Imaging](imaging-guide.md) · [Measurements](measurements-guide.md)

Use an Apple Silicon Mac or Ubuntu 24.04+ with Python 3.12 or newer. The desktop
viewer needs a graphical session. The TOPDON TC002C Duo (`2bdf:0102`) is the
physically implemented camera; `virtual-dual` is available for hardware-free exploration.
Other Linux NVIDIA hosts and physical dual-camera combinations have the limits
described in the [backend documentation](camera-backends.md).

## macOS

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
If you already use Homebrew, `brew install uv` is one supported installation method.
Then run in a terminal:

```sh
git clone https://github.com/manbehindthemadness/based-thermals.git
cd based-thermals
uv sync --no-dev
.venv/bin/based-thermals-desktop --rotate 90
```

uv creates the local environment and resolves the project's Python requirement.
Launch as your normal user. The Duo USB helper asks for sudo authorization in
that terminal; the UI, model processing, downloads and saved files stay unprivileged.
Close other applications that own the camera before starting.

If an IDE launch cannot display the password prompt, launch from a terminal instead.
You can run `sudo -v` there first, then launch from that same terminal. This does
not grant permanent or passwordless access. See [macOS USB authorization](macos-usb.md)
for the trust boundary and targeted repair of files owned by earlier root launches.

## Ubuntu 24.04 and later

Install the system libraries and create the environment. These apt commands are
for Ubuntu, not package-manager instructions for every Linux distribution:

```sh
sudo apt update
sudo apt install git python3-venv libusb-1.0-0 libgl1 libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0 zenity
git clone https://github.com/manbehindthemadness/based-thermals.git
cd based-thermals
python3 -m venv .venv
.venv/bin/python -m pip install uv
.venv/bin/uv sync --inexact --no-dev
```

This uses Ubuntu's Python; check that it is 3.12 or newer. If uv is already on
PATH, `uv sync --no-dev` is also supported. `zenity` supplies the Save dialog.

The environment-local bootstrap uses `--inexact` so syncing does not remove the
uv installer itself. uv normally removes packages outside the lockfile during
an exact sync; see [uv's sync behavior](https://docs.astral.sh/uv/concepts/projects/sync/).
In other guides, commands beginning with `uv` assume it is on PATH; substitute
`.venv/bin/uv` when you installed it only inside this environment.
For those environment-local `sync` commands, also include `--inexact` to retain uv.

Install the included rule once to give the active local desktop user access to
the Duo without running the application as root:

```sh
sudo install -m 644 packaging/udev/70-topdon-duo.rules /etc/udev/rules.d/70-topdon-duo.rules
sudo udevadm control --reload-rules
```

Unplug and reconnect the camera after installing the rule, then launch normally:

```sh
.venv/bin/based-thermals --diagnose
.venv/bin/based-thermals-desktop --rotate 90
```

The rule is specific to `2bdf:0102` and the active local desktop session.
Remote/headless users and other devices need their own appropriate USB permissions.
The Linux capture path temporarily detaches this camera's `uvcvideo` interfaces
and reattaches them on close. Do not run two camera owners concurrently.

## First launch and GPU setup

Runtime setup happens before camera construction. It retains a working runtime,
or stages and verifies a replacement in the application cache. Apple uses CoreML;
Linux NVIDIA selection checks CUDA capabilities, with an exact Orin compatibility
exception for its verified build. Generic CUDA 12/13 choices remain available to
other NVIDIA machines. Wheel availability alone does not prove usable GPU kernels.

Initial setup can take time and download several GB of native libraries. Progress
appears in the terminal before the viewer opens. Setup does not install drivers,
change system libraries, or change USB permissions. Optional setup failures leave
the installed CPU fallback available. GPU execution controls reflect capabilities;
choose the backend in the applicable pipeline node.

Normal viewer launches can use `.venv/bin/` commands or `uv run`; automatic runtime
caches survive dependency sync. Old advice to always use `uv run --no-sync` applies
to manually replaced environment packages, not the separate automatic cache.
Exact versions, tested hosts and managed-install details are in
[runtime setup](onnx-runtime-setup.md), [Linux CUDA](nvidia-linux.md),
[Jetson/Orin](nvidia-jetpack.md) and [DGX Spark](nvidia-spark.md).

To retry setup without opening the camera:

```sh
.venv/bin/python -m topdon_duo.runtime_setup --retry
```

For an explicitly offline or managed launch that uses only the environment's runtime:

```sh
BASED_THERMALS_RUNTIME_SETUP=0 .venv/bin/based-thermals-desktop
```

This bypass skips automatic cache activation too. AI model downloads are separate
from runtime provisioning; see [visual models](visual-models.md).

## Try the UI without hardware

```sh
.venv/bin/based-thermals-desktop --camera virtual-dual
.venv/bin/based-thermals-desktop --camera virtual-dual --image-source visible
```

These use synthetic thermal/visible streams without USB authorization. They help
explore layout, measurements and pipelines, but do not validate physical hardware.

## Update an existing checkout

Close the viewer and protect local changes before updating the checkout with Git.
Run `uv sync --no-dev` afterward (`.venv/bin/uv sync --inexact --no-dev` for environment-local uv).
Use `--dev` when you need development tools. Reopen the viewer normally.
Keep your existing project directory and IDE configuration; no rename is required.
The previous `topdon-duo`, `topdon-duo-desktop` and `topdon-duo-models` commands remain aliases.

The based-gan example is optional and lives in its own repository. To populate it
when working with that example, run from the repository root:

```sh
git submodule update --init examples/custom_nodes/based-gan
```

It is not required for normal capture or other examples. Check its own requirements
before downloading/converting weights or loading its Custom package.

## Preferences, models and disk space

Existing data uses the legacy `topdon-duo` directory names:

- Preferences/presets: `~/Library/Application Support/topdon-duo` on macOS;
  `$XDG_CONFIG_HOME/topdon-duo` (default `~/.config/topdon-duo`) on Linux.
- Downloaded models: `~/Library/Caches/topdon-duo/models` on macOS;
  `$XDG_CACHE_HOME/topdon-duo/models` (default `~/.cache/topdon-duo/models`) on Linux.
- Native runtimes: the corresponding cache root's `topdon-duo/runtimes` directory.

`BASED_THERMALS_MODEL_DIR` and `BASED_THERMALS_RUNTIME_DIR` override the respective
caches; the legacy `TOPDON_MODEL_DIR` remains supported for models. Model-cache
clearing is available from the Imaging preset menu and preserves protected/local
artifacts. It does not clear native runtime packages. See [Imaging](imaging-guide.md#manage-model-downloads).

## Troubleshooting

| Symptom                       | Next step                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------------ |
| `uv` not found                | Install uv or use `.venv/bin/uv` from the Ubuntu bootstrap                                 |
| Camera listed but cannot open | Close other camera apps; verify permissions and reconnect                                  |
| macOS authorization fails     | Launch in a terminal that can display the USB helper's prompt                              |
| Linux permission denied       | Install the device rule, reconnect, and use the active local desktop session               |
| Qt/display fails on Linux     | Check the system libraries above and the graphical session                                 |
| GPU unavailable               | Read the startup reason; run runtime setup with `--retry`; consult the host-specific notes |
| Temperatures held or `--`     | Wait for a valid frame after shutter calibration/startup; read the status bar              |
| View is too large             | Launch with `--reset-window-size`; use Rotate for layout and orientation                   |

USB discovery is read-only and does not run runtime installation:

```sh
.venv/bin/based-thermals --diagnose
```

For detailed startup/frame traces, use
`.venv/bin/based-thermals-desktop --diagnostics diagnostics/viewer.jsonl`.
See [development and diagnostics](development.md) for interpreting traces and
monitoring an existing Linux viewer without opening the camera again.
