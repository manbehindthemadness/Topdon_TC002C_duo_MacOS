"""
Desktop labels, toolbar dimensions, and recording defaults.
"""

import logging

from ..camera import FRAME_RATE

LOG = logging.getLogger(__name__)
WINDOW_NAME = "TOPDON TC002C Duo"
TOOLBAR_BUTTON_HEIGHT = 24
TOOLBAR_PADDING = 4
TOOLBAR_FONT_SCALE = 0.36
TIMELAPSE_DEFAULT_FPM = 60
TIMELAPSE_MAX_FPM = FRAME_RATE * 60
SAVE_DIALOG_SCRIPT = """
on run argv
    tell current application to activate
    set defaultName to item 1 of argv
    set defaultFolder to item 2 of argv
    if defaultFolder is "" then
        set chosenFile to choose file name with prompt "Save thermal capture" default name defaultName
    else
        set chosenFile to choose file name with prompt "Save thermal capture" default name defaultName default location (POSIX file defaultFolder)
    end if
    return POSIX path of chosenFile
end run
"""
