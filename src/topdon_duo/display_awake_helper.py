"""Hold Linux idle inhibition separately from OpenCV until the parent closes stdin."""

import os
import selectors
import signal
import subprocess
import sys

from .display_awake import REASON


def gnome_main(executable):
    # Hold GNOME's inhibit-only process and watch our stdin for parent lifetime.
    signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
    process = subprocess.Popen(
        [
            executable,
            "--inhibit",
            "idle",
            "--inhibit-only",
            "--app-id",
            "topdon-duo",
            "--reason",
            REASON,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(sys.stdin, selectors.EVENT_READ)
            while True:
                if selector.select(timeout=0.2) and not os.read(sys.stdin.fileno(), 4096):
                    return 0
                if process.poll() is not None:
                    print("Desktop idle inhibitor stopped unexpectedly", file=sys.stderr)
                    return 2
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)


def dbus_main():
    from PySide6.QtCore import QCoreApplication, QSocketNotifier
    from PySide6.QtDBus import QDBusConnection, QDBusInterface, QDBusMessage

    app = QCoreApplication(sys.argv)
    bus = QDBusConnection.sessionBus()
    interface = QDBusInterface(
        "org.freedesktop.ScreenSaver", "/ScreenSaver", "org.freedesktop.ScreenSaver", bus
    )
    interface.setTimeout(2000)
    reply = interface.call("Inhibit", "topdon-duo", REASON)
    if reply.type() == QDBusMessage.ErrorMessage:
        print(reply.errorMessage(), file=sys.stderr)
        return 2
    # The connection owns the inhibitor; disconnecting releases it on exit.
    notifier = QSocketNotifier(sys.stdin.fileno(), QSocketNotifier.Read)
    notifier.activated.connect(
        lambda *_args: app.quit() if not os.read(sys.stdin.fileno(), 4096) else None
    )
    return app.exec()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--gnome":
        return gnome_main(sys.argv[2])
    return dbus_main()


if __name__ == "__main__":
    raise SystemExit(main())
