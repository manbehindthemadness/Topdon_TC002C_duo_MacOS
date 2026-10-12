# Temperature accuracy and calibration

[Documentation home](../README.md) · [Measurements guide](measurements-guide.md)

## Temperature accuracy

The desktop uses camera-corrected counts converted with `raw / 64 - 50` for
statistics, cursor readings, sample spots and saved measurements. The **Imaging**
popup's ambient row is the sole desktop ambient control. Its effective hardware
value also appears in the readout and capture metadata; if configuration cannot
be read, ambient is shown as unavailable. The old desktop `--ambient` argument
is accepted for compatibility but ignored with a warning.

Live hardware experiments found that ambient temperature, distance and emissivity
can be set on the camera, and mode-8 counts usually agree closely with its
telemetry. Restoring settings keeps the camera conversion and original hardware
ambient value. Readings remain approximate and are not measurement-grade.

The web viewer retains its software cold-background anchor and `--ambient` option.

## Hardware research

[Hardware investigation notes](hardware-research.txt) record over 60 live
control trials, the verified USB protocol, preserved settings, six captured
stream formats, processing controls, and unresolved features. The official
Android app's TC002C Duo TISR switch was traced to host-side enhancement using
Anime4KCPP's rule-based Anime4K09 algorithm. ACNet CNN models are also bundled,
but use a separate constructor. The generic SDK's ignored super-resolution field
is a separate interface. The app's iOS implementation has not been inspected.
[Enhancement source and model notes](anime-enhancement.txt) record the
obtained upstream code, verified ACNet weights, and a successful local test.

Large captures, original configuration payloads and extracted research material
remain local under the gitignored `diagnostics/telemetry-research/` directory.

## Calibrate with household objects

Open **Imaging** from the main window to find the calibration controls:

- **Post-it → distance:** use a standard 3 × 3 inch note (nominally 76 × 76 mm;
  measure yours). Hold it flat in front of a warm palm so its cooler square has
  visible edges. Enter its size and a tape-measured distance from the front lens,
  detect it, then save the reference. At another distance, measure the same note
  and apply the estimated distance to the camera.
- **Shiny spoon → reflected temperature:** show the reflector target, cover the
  entire center circle with shiny bare metal, and measure while holding steady.
  A stable reading saves automatically. Apply the saved reflected temperature to
  update the camera. This experimental method depends on spoon orientation;
  compare with a crumpled then flattened foil reflector before relying on it.
- **Known surface temperature → emissivity:** select a point and enter an
  independently measured surface temperature, fit emissivity, then apply it.

All three saved calibration references load automatically when the program
starts and appear in their Camera sections. Values you have **applied** to the
camera also restore automatically after connection. Saving a reference alone
keeps it available for later use; it does not apply a new camera setting. For
Post-it distance, the saved reference is reused for new measurements rather than
reusing an unfinished distance estimate. Reset display and Restore camera
settings preserve saved calibration references; Restore camera settings clears
applied hardware preferences until you apply them again. Repeat calibration when
the camera, target material, or relevant measurement conditions change.

## Distance calibration with a square

Imaging → Distance calibration uses a **76 × 76 mm Post-it** by default. You can
edit the square's side length. Keep it flat, steady and facing the camera near
the image center. Hold the cooler note in front of your warm palm, with skin
visible around its edges. The sticky edge alone may not hold the whole note flat.
The note needs thermal contrast against the hand; if it warms to the same
temperature, use a fresh note or let it cool before trying again.

1. Measure the distance from the camera's front lens to the note. Enter that
   reference distance and the side length in centimeters or inches, according to
   the selected measurement units.
2. Click **Detect reference square** and hold steady. The detected outline appears
   on the main image and locks after eight consistent frames. Click **Save reference**.
3. Move the same square to another distance, keeping it face-on. Click
   **Measure square** and hold steady for another automatic detection.
4. Check the estimated distance against a tape measurement at another distance.
   **Apply distance to camera** writes the estimate to the hardware distance
setting, rounded to its 1 cm precision. Detecting and saving do not write it.

The reference persists across restarts; incomplete selections and estimates do
not. Recalibrate after changing the camera or lens. Clearing the calibration
leaves the hardware distance setting unchanged. Controls are locked while logging.

This is an approximate, single-reference range estimate: distance equals the
reference distance multiplied by reference pixel width divided by current pixel
width. It averages the four edges in native thermal coordinates, so display
scaling, AI upsampling, mirroring and rotation do not change the scale. A new
rotation or mirror change cancels the current detection. Crossed corners,
noticeably tilted or uneven squares, and edges below 12 native pixels are
rejected. Estimates outside the hardware's 0.3–99 m range cannot be applied.
A 76 mm target may become too small well before the camera's maximum range.

The initial shape detector sweeps temperature thresholds for square contours,
also joins straight edges when uneven warming breaks the contour, refines the
corners, and checks that warmer surroundings support at least three
edges. Its starting hand-background range is 24–40 °C, with at least 0.75 °C
contrast. It uses the native Celsius plane, independent of display palette,
units and AI enhancement. Missing or multiple qualifying squares keep detection
waiting; moving or losing the target restarts the stability check. These are
thermal heuristics, so other cool squares against warm backgrounds can qualify.
Check the displayed outline before saving or applying a measurement.

The method follows the [pinhole projection model documented by OpenCV](https://docs.opencv.org/5.0/main_modules/calib.html).
It does not correct lens distortion or solve target tilt; a single reference is
not a full camera calibration. The front-lens measurement also approximates the
optical origin. Verify accuracy at additional measured distances before relying
on it for temperature compensation.

## Emissivity calibration with a known temperature

Imaging → Emissivity calibration fits the camera's global emissivity setting to
an independently measured surface temperature:

1. Click **Select reference point**. Measuring-spot, cursor and distance-detection
   markers disappear. Click the reference surface in the main thermal image;
   its own reference marker remains visible with a live temperature label.
2. The selected pixel's current temperature is copied into **Known surface
   temperature**, in the selected °C or °F units. Edit it to your independently
   measured temperature. Live updates preserve your edit. Selecting a new point
   imports its temperature again.
3. Click **Fit emissivity** and keep the camera and reference steady. The fit
   temporarily probes the camera's supported 0.01–1.00 emissivity range, waits
   one second after each write, and collects five stable readings. Other settings
   are locked during fitting, and temperature logging cannot start.
4. A successful fit saves the calibration separately, restores the prior
   emissivity, clears the reference marker and offers the fitted value.
   **Apply emissivity to camera** keeps it as a saved hardware preference and
   restores the markers. **Close calibration** ends point selection and restores
   the markers while retaining any saved fit. Closing during fitting also restores
   the previous setting.

**Reset display settings** and **Restore camera settings** preserve both the
distance reference and the saved emissivity calibration, including across
restarts. Restore returns hardware settings to their original values; use
**Apply saved emissivity** to reapply the retained fit when appropriate.

The fit reads the native camera-corrected plane rather than display smoothing,
palette pixels or AI output. It tests the observed response rather than assuming
a radiometric formula. It stops without a result for weak or inconsistent
response, an unreachable temperature, unstable readings, a timeout, or when the
camera's 0.01 emissivity steps cannot match within 0.3 °C. This matching tolerance
is not a claim about the camera's absolute temperature accuracy.

Use an independent surface-temperature reference and appropriate ambient,
reflected-temperature and distance settings. The fitted emissivity is specific
to the selected material and conditions, while applying it affects the entire
image. The procedure follows the known-temperature adjustment approach in
[FLIR's thermographic measurement guidance](https://support.flir.com/docdownload/assets/web/4jau/en-us/T505000.xml.html).
The workflow and feedback fit are covered by simulated-camera tests; a physical
trial with a measured reference is still required for this camera.

## Reflected-temperature calibration

Imaging → Reflected-temperature calibration uses a small circle at the center
of the image. Show the target, cover the entire circle with a shiny bare-metal
spoon or foil reflector near the subject, and click **Measure stable temperature**.
Other measurement markers are hidden during calibration. The reading uses the
median of native temperature pixels inside the circle, with a one-second settling
period and at least two seconds of stability within 0.3 °C. It times out after
30 seconds if the reading does not settle.

Measurement temporarily sets emissivity to 1.00 and optical transmission to 100%,
then restores their previous values and override states on success, cancellation,
or failure. Other environmental corrections remain at their configured values.
A stable result is saved independently and clears the circle. **Apply saved
reflected temperature** writes it to the camera at its 0.1 °C precision. The saved
reference survives restarting, Reset display, and Restore camera settings; the
Imaging panel displays it in the selected Celsius/Fahrenheit unit. Settings and
logging are locked during measurement.

A spoon is an experimental reflector: curvature makes its reading depend on
orientation, and it may reflect the operator or camera. Compare it against the
[FLIR foil reflector procedure](https://support.flir.com/docdownload/assets/web/2p5q/en-us/T505000.xml.html)
(crumpled then flattened aluminum foil) before relying on it. A steady reading
alone cannot establish that a material is reflective or that the calibration is
accurate. This camera's distance and atmospheric corrections still need physical
validation for this procedure.

## Automatic controls and future calibrations

Planned Auto behavior: adapt brightness, contrast, general/spatial/temporal noise
reduction and detail enhancement using image histograms, estimated noise, motion
and edge strength. These mappings still need validation on this camera.
Ambient temperature stays manual and is excluded from automatic calculation.
Distance uses automatic square detection with a measured reference. Emissivity
has the known-temperature fitting workflow above; both still need accuracy
validation against physical references. Reflected temperature has an experimental reflector workflow; optical
transmission remains manual until a suitable reference-based calibration is available. Palettes,
mirrors and overlays remain user preferences. Weather humidity is a possible
optional outdoor estimate, with its source and age shown; it is not a measured
indoor value. Automatic calculation remains a future feature.

| Future calibration             | Required reference or data                                                                                                           | Proposed method                                                                        | Validation / limits                                                                                                             |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Brightness and contrast        | Fixed scenes with narrow and wide temperature spans                                                                                  | Map histogram percentiles and clipping to hardware settings; smooth changes over time  | Check clipping and stability; distinguish existing camera gain adjustment from host adjustment                                  |
| General and spatial denoising  | Repeated frames of flat regions and fine-detail targets                                                                              | Estimate noise and measure the effect of each strength                                 | Compare noise reduction against lost detail; scene texture must not count as noise                                              |
| Temporal denoising             | Static scenes and controlled target motion                                                                                           | Map temporal noise and motion estimates to filtering strength                          | Check trails and response delay; allow hardware settling before comparing frames                                                |
| Detail enhancement             | Edge targets captured at several noise levels                                                                                        | Select strength using edge clarity and noise estimates                                 | Check halos and amplified noise; enhanced pixels do not establish new measurement resolution                                    |
| Distance / camera geometry     | Known-size target, measured distances and target orientation                                                                         | Calibrate projection and estimate distance from target size in native pixels           | Validate at held-out distances; arbitrary objects and unknown orientation do not give reliable absolute distance                |
| Emissivity                     | Known-emissivity reference on the same surface or an independent surface-temperature measurement; supplied environmental corrections | Fit emissivity to the reference temperature, then restore original settings            | Verify thermal equilibrium and useful temperature contrast; result applies to that material, surface and measurement conditions |
| Reflected apparent temperature | Suitable reflector placed near the target                                                                                            | Guided reflector measurement with a documented temporary camera configuration          | Validate the procedure against this camera's supported distance and correction settings; restore originals afterward            |
| Optical transmission           | Known IR-transmitting optics and a stable reference target                                                                           | Compare reference measurements with and without the optics under controlled conditions | Confirm the hardware field's meaning and compensation model before fitting; a single temperature ratio is insufficient          |

For later trials, retain original hardware payloads, reference conditions, camera
and firmware identification, raw frames, applied settings, settling time and
validation results. Record estimates separately from measured references and
preserve the camera's corrected temperature conversion throughout display tuning.
The measurement reference procedures are informed by
[FLIR's thermographic measurement guidance](https://support.flir.com/docdownload/assets/web/4jau/en-us/T505000.xml.html);
they still require validation for this camera.

While the desktop viewer holds a camera connection, it requests temporary display
idle inhibition to prevent monitor blanking from stalling the display loop. The
request is released on exit, including after a camera error. On GNOME this uses
`gnome-session-inhibit --inhibit idle`; other Linux desktops use the ScreenSaver
D-Bus interface, and macOS uses `caffeinate -d`. System power settings are unchanged.
If the desktop refuses the request, the viewer reports the failure.

After reconnecting, the camera may stream before its settings interface is ready.
If initial settings reads fail, the viewer retries them as frames arrive, at most
once per second for 30 seconds. Saved hardware values and calibration commands
are applied only after the complete original settings have been read and saved.
This lets unavailable controls recover without repeated manual restarts.

Desktop USB acquisition runs continuously on a dedicated thread, retaining the
newest complete frame for the UI. Rendering does not stop the camera reads or
build a queue of old images. If no new frame arrives for half a second, the image
and readouts are held while graphs mark measurements invalid and CSV logging
skips them. Window events remain responsive while capture waits for the camera.

On macOS and Linux, camera capture keeps 32 asynchronous USB reads queued so host
scheduling delays do not interrupt capture requests. Completed reads are
resubmitted before their packets are processed. Use `--usb-queue-depth 0` for
diagnostic synchronous capture, or choose a depth from 2 to 128.
