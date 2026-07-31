---
name: piY Web
description: A local engineering signal desk for the piY coding agent
colors:
  cool-paper: "#eef3f1"
  work-surface: "#f8fbfa"
  raised-surface: "#ffffff"
  active-surface: "#d8e6e2"
  graphite-ink: "#14201d"
  secondary-ink: "#465650"
  quiet-ink: "#71817b"
  rule: "#c4d0cc"
  live-signal: "#008f8a"
  live-signal-soft: "#c8e9e5"
  pending-signal: "#b47d00"
  danger-signal: "#c64538"
  dark-field: "#101614"
  instrument-field: "#15211e"
  instrument-grid: "#2d423b"
  instrument-trace: "#35c7c0"
  instrument-label: "#77d6cf"
typography:
  brand:
    fontFamily: "Georgia, serif"
    fontSize: "19px"
    fontWeight: 400
    lineHeight: 1
    letterSpacing: "0"
  headline:
    fontFamily: "Aptos, Segoe UI Variable, Segoe UI, Arial, sans-serif"
    fontSize: "19px"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "0"
  subheading:
    fontFamily: "Aptos, Segoe UI Variable, Segoe UI, Arial, sans-serif"
    fontSize: "17px"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "0"
  section:
    fontFamily: "Aptos, Segoe UI Variable, Segoe UI, Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "0"
  title:
    fontFamily: "Aptos, Segoe UI Variable, Segoe UI, Arial, sans-serif"
    fontSize: "14px"
    fontWeight: 700
    lineHeight: 1.25
    letterSpacing: "0"
  body:
    fontFamily: "Aptos, Segoe UI Variable, Segoe UI, Arial, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.65
    letterSpacing: "0"
  label:
    fontFamily: "Cascadia Code, SFMono-Regular, Consolas, monospace"
    fontSize: "10px"
    fontWeight: 650
    lineHeight: 1.5
    letterSpacing: "0"
  small:
    fontFamily: "Aptos, Segoe UI Variable, Segoe UI, Arial, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "0"
  compact:
    fontFamily: "Aptos, Segoe UI Variable, Segoe UI, Arial, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "0"
  data:
    fontFamily: "Cascadia Code, SFMono-Regular, Consolas, monospace"
    fontSize: "11px"
    fontWeight: 400
    lineHeight: 1.55
    letterSpacing: "0"
  micro:
    fontFamily: "Cascadia Code, SFMono-Regular, Consolas, monospace"
    fontSize: "9px"
    fontWeight: 650
    lineHeight: 1.5
    letterSpacing: "0"
  nano:
    fontFamily: "Cascadia Code, SFMono-Regular, Consolas, monospace"
    fontSize: "8px"
    fontWeight: 650
    lineHeight: 1.5
    letterSpacing: "0"
rounded:
  hairline: "1px"
  instrument: "3px"
  control: "4px"
  framed: "5px"
  bubble: "6px"
  panel: "7px"
spacing:
  tight: "6px"
  control: "8px"
  group: "14px"
  section: "20px"
components:
  button-primary:
    backgroundColor: "{colors.live-signal}"
    textColor: "{colors.raised-surface}"
    rounded: "{rounded.control}"
    height: "38px"
    width: "38px"
  input:
    backgroundColor: "{colors.raised-surface}"
    textColor: "{colors.graphite-ink}"
    rounded: "{rounded.panel}"
    padding: "8px 13px"
  instrument-panel:
    backgroundColor: "{colors.instrument-field}"
    textColor: "{colors.live-signal-soft}"
    rounded: "{rounded.instrument}"
    padding: "9px"
---

# Design System: piY Web

## Overview

**Creative North Star: "The Engineering Signal Desk"**

piY Web treats the coding agent as a local instrument under active observation. The interface is compact, operational, and precise: the conversation occupies the work floor while sessions, files, models, and run telemetry stay on fixed perimeter rails. Brand character comes from measured rules and readable state signals rather than decorative technology motifs.

The system remains quiet during reading and becomes visibly active only when the agent runs. Its visual anti-reference is the generic rounded chat dashboard: controls stay compact, messages are unframed, and state color always carries meaning.

**Key Characteristics:**

- Cool paper and graphite fields support long work sessions in light and dark environments.
- Cyan marks live or selected state, yellow marks pending work, and red marks failure.
- One-pixel rules establish structure; shadows are reserved for drawers, toasts, and the composer.
- Monospace is limited to code, paths, measurements, state, and instrument labels.

## Colors

The palette combines low-chroma green neutrals with three explicit signal colors.

### Primary

- **Live Cyan:** used for active controls, selected nodes, running traces, and focus state.

### Secondary

- **Pending Amber:** used for work in progress, directories, and attention that is not an error.
- **Fault Red:** reserved for failed tools, invalid requests, destructive actions, and abort state.

### Neutral

- **Cool Paper:** the central work field in light mode.
- **Raised White:** navigation rails, controls, and the top instrument bar.
- **Graphite Ink:** primary text and the dark user-message surface.
- **Instrument Field:** code blocks and the oscilloscope-style run trace.

**The Signal Integrity Rule.** Cyan, amber, and red are state channels, never decorative accents.

## Typography

**Body Font:** Aptos with Segoe UI and Arial fallbacks  
**Label/Mono Font:** Cascadia Code with system monospace fallbacks
**Brand Symbol Font:** Georgia for the pi glyph only

**Character:** The UI face is neutral and compact. Monospace acts as measured data, preserving the distinction between human conversation and machine state.

### Hierarchy

- **Title** (700, 14px, 1.25): channel, inspector, and compact panel titles.
- **Body** (400, 14px, 1.65): chat messages and readable content, constrained to approximately 72 characters when possible.
- **Label** (650, 10px, 1.5): paths, counters, statuses, and instrument metadata.
- **Micro scale** (8-13px): compact controls, file kinds, timestamps, and numeric readouts; never primary prose.

**The Measured Type Rule.** Uppercase and monospace are used only where the content behaves like a label or measurement.

## Layout

Desktop uses a fixed 278px left tool rail, a flexible central work floor, and a 362px inspector. At 920px the inspector becomes a right drawer. At 700px the navigation becomes a left drawer, the inspector becomes full width, controls use at least 44px touch targets, and the conversation remains the only persistent surface.

The top bar is 50px high. Content groups use a compact 6-8px internal rhythm and 14-20px separation between distinct functions. Chat content is limited to 880px for scanning and code readability.

**The Clear Floor Rule.** The central column belongs to the current task; navigation, file inspection, and telemetry remain on perimeter rails.

## Elevation & Depth

The system is flat by default. One-pixel rules and tonal surface changes define hierarchy. A soft offset shadow appears only on temporary layers such as mobile drawers and toasts, and on the composer to mark the active input plane.

### Shadow Vocabulary

- **Temporary Layer** (`0 12px 32px rgba(18, 31, 27, 0.16)`): drawers and toasts.
- **Input Plane** (`0 7px 22px rgba(18, 31, 27, 0.1)`): the composer at rest and focus.

**The Flat-at-Rest Rule.** Persistent panels use borders or tonal separation, not shadows.

## Shapes

Controls use small 3-4px corners; the composer and framed tool output use 5-7px corners. Square nodes, state lights, and the piY mark reinforce the instrument vocabulary. Pills and oversized soft containers are not part of the system.

## Components

### Buttons

- **Shape:** compact controls use 4-5px corners and square icon geometry.
- **Primary:** live cyan with white icon; hover shifts to graphite.
- **Hover / Focus:** border or tonal state plus a three-pixel translucent cyan focus ring.
- **Mobile:** primary and navigation controls expand to at least 44px.

### Inputs / Fields

- **Style:** raised surface, one-pixel rule, 4-7px corners.
- **Focus:** rule changes to live cyan and receives the shared focus ring.
- **Error / Disabled:** errors use fault red; disabled controls remain structurally visible at reduced opacity.

### Navigation

The left rail uses segmented session/file tabs, measured section headers, and full-width list rows. Active rows use a quiet surface change, a one-pixel cyan index, and a filled square node. Mobile navigation becomes an off-canvas rail with explicit scrim dismissal.

### Signal Scope

The run inspector uses a fixed dark field, cyan trace, subtle measurement grid, and tabular state rows. Trace motion appears only while the agent is running and stops under reduced-motion preferences.

### Composer

The composer is a stable bottom work plane with a bounded growing textarea, numeric character readout, and one icon command that changes from send to stop while streaming.

## Do's and Don'ts

### Do:

- **Do** keep chat content unframed and readable while using frames for real tool or code boundaries.
- **Do** map every signal color to an observable state.
- **Do** keep desktop information dense and mobile controls touch-safe.
- **Do** expose focus, loading, empty, running, failed, and disabled states.

### Don't:

- **Don't** turn the interface into a generic field of rounded cards.
- **Don't** use monospace for ordinary prose or brand display text.
- **Don't** add gradients, glow effects, or decorative data graphics.
- **Don't** hide the current task behind navigation or telemetry on narrow screens.
